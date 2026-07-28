use std::collections::{BTreeMap, HashMap};

use crate::GlmrtError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionRoutePlanEntry {
    pub row_index: usize,
    pub expert_id: usize,
    pub intermediate_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionFirstRouteGroup {
    pub route_indices: Vec<usize>,
    pub completed_rows: Vec<usize>,
    pub ready_after_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionFirstRoutePlan {
    pub groups: Vec<CompletionFirstRouteGroup>,
    pub activation_row_order: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RollingExpertRowPackConfig {
    pub logical_chunk_rows: usize,
    pub max_pack_rows: usize,
    pub lookahead_rows: usize,
    pub expert_tile_rows: usize,
    pub selection_quantum_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollingExpertRowPackPlan {
    pub packs: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollingExpertRowPackEmission {
    pub row_indices: Vec<usize>,
    pub emitted_pack_index: usize,
    pub admitted_rows: usize,
    pub oldest_pending_row: usize,
    pub max_selected_row_offset: usize,
    pub deadline_row_exclusive: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingRollingExpertRow {
    row_index: usize,
    groups: Vec<usize>,
}

/// Rolling window scaffold. Callers feed routed source chunks and receive every
/// physical pack made eligible by that admission; source and physical sizes are
/// intentionally independent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollingExpertRowPackAccumulator {
    config: RollingExpertRowPackConfig,
    pending: Vec<PendingRollingExpertRow>,
    group_indices: HashMap<(usize, usize), usize>,
    admitted_rows: usize,
    emitted_packs: usize,
    finished: bool,
}

impl RollingExpertRowPackAccumulator {
    pub fn new(config: RollingExpertRowPackConfig) -> Result<Self, GlmrtError> {
        validate_rolling_expert_row_pack_config(config)?;
        Ok(Self {
            config,
            pending: Vec::with_capacity(config.lookahead_rows),
            group_indices: HashMap::new(),
            admitted_rows: 0,
            emitted_packs: 0,
            finished: false,
        })
    }

    pub fn push_chunk(
        &mut self,
        entries: &[CompletionRoutePlanEntry],
        row_count: usize,
    ) -> Result<Vec<RollingExpertRowPackEmission>, GlmrtError> {
        ensure(!self.finished, "rolling row pack accumulator is finished")?;
        ensure(row_count > 0, "rolling row pack chunk must contain rows")?;
        ensure(
            row_count <= self.config.lookahead_rows,
            format!(
                "rolling row pack admission has {row_count} rows but lookahead capacity is {}",
                self.config.lookahead_rows
            ),
        )?;
        let row_start = self.admitted_rows;
        let row_end = row_start
            .checked_add(row_count)
            .ok_or_else(|| rejected("rolling row pack admitted row count overflow"))?;
        let mut rows = vec![Vec::<(usize, usize)>::new(); row_count];
        for entry in entries {
            ensure(
                (row_start..row_end).contains(&entry.row_index),
                format!(
                    "rolling row pack chunk route row {} is outside [{row_start}, {row_end})",
                    entry.row_index
                ),
            )?;
            let row = &mut rows[entry.row_index - row_start];
            let group = (entry.expert_id, entry.intermediate_rows);
            ensure(
                !row.contains(&group),
                format!(
                    "rolling row pack row {} repeats expert group {:?}",
                    entry.row_index, group
                ),
            )?;
            row.push(group);
        }
        for (row_index, row) in rows.iter().enumerate() {
            ensure(
                !row.is_empty(),
                format!(
                    "rolling row pack row {} has no routes",
                    row_start + row_index
                ),
            )?;
        }
        let mut pending_rows = Vec::with_capacity(row_count);
        for (row_offset, groups) in rows.into_iter().enumerate() {
            let mut group_ids = Vec::with_capacity(groups.len());
            for group in groups {
                let next_group_index = self.group_indices.len();
                group_ids.push(*self.group_indices.entry(group).or_insert(next_group_index));
            }
            pending_rows.push(PendingRollingExpertRow {
                row_index: row_start + row_offset,
                groups: group_ids,
            });
        }
        self.pending.extend(pending_rows);
        self.admitted_rows = row_end;
        let emissions = self.emit_ready(false)?;
        ensure(
            self.pending.len() < self.config.lookahead_rows,
            "rolling row pack admission did not restore the live-window bound",
        )?;
        if let Some(deadline) = self
            .admitted_rows
            .checked_sub(self.config.lookahead_rows)
            .filter(|deadline| *deadline > 0)
        {
            ensure(
                self.pending
                    .first()
                    .is_none_or(|row| row.row_index >= deadline),
                format!("rolling row pack retained a row older than admission deadline {deadline}"),
            )?;
        }
        Ok(emissions)
    }

    pub fn finish(&mut self) -> Result<Vec<RollingExpertRowPackEmission>, GlmrtError> {
        ensure(!self.finished, "rolling row pack accumulator is finished")?;
        self.finished = true;
        self.emit_ready(true)
    }

    pub fn pending_rows(&self) -> usize {
        self.pending.len()
    }

    pub fn admitted_rows(&self) -> usize {
        self.admitted_rows
    }

    fn emit_ready(
        &mut self,
        draining: bool,
    ) -> Result<Vec<RollingExpertRowPackEmission>, GlmrtError> {
        let mut emissions = Vec::new();
        while !self.pending.is_empty() {
            let deadline_requires_emission = self
                .admitted_rows
                .checked_sub(self.config.lookahead_rows)
                .filter(|deadline| *deadline > 0)
                .is_some_and(|deadline| self.pending[0].row_index < deadline);
            if !draining
                && self.pending.len() < self.config.lookahead_rows
                && !deadline_requires_emission
            {
                break;
            }
            emissions.push(self.emit_one()?);
        }
        Ok(emissions)
    }

    fn emit_one(&mut self) -> Result<RollingExpertRowPackEmission, GlmrtError> {
        let oldest_pending_row = self.pending[0].row_index;
        let candidate_row_end = oldest_pending_row.saturating_add(self.config.lookahead_rows);
        let candidate_rows = self
            .pending
            .iter()
            .take_while(|row| row.row_index < candidate_row_end)
            .count();
        let capacity = self.config.max_pack_rows.min(candidate_rows);
        ensure(
            self.pending.len() < self.config.max_pack_rows || capacity == self.config.max_pack_rows,
            format!(
                "rolling row pack live window exposed only {capacity} of {} physical rows",
                self.config.max_pack_rows
            ),
        )?;
        let deadline_row_exclusive = self
            .admitted_rows
            .checked_sub(self.config.lookahead_rows)
            .filter(|deadline| *deadline > 0);
        let deadline_rows = deadline_row_exclusive
            .map(|deadline| {
                self.pending
                    .iter()
                    .take_while(|row| row.row_index < deadline)
                    .count()
            })
            .unwrap_or(0);
        let required_deadline_rows = deadline_rows.min(capacity);

        let mut row_group_offsets = Vec::with_capacity(candidate_rows + 1);
        let mut row_groups = Vec::with_capacity(candidate_rows.saturating_mul(8));
        row_group_offsets.push(0);
        for row in self.pending.iter().take(candidate_rows) {
            row_groups.extend_from_slice(&row.groups);
            row_group_offsets.push(row_groups.len());
        }
        let local_config = RollingExpertRowPackConfig {
            logical_chunk_rows: self
                .config
                .logical_chunk_rows
                .min(capacity)
                .max(required_deadline_rows),
            max_pack_rows: capacity,
            lookahead_rows: candidate_rows,
            expert_tile_rows: self.config.expert_tile_rows,
            selection_quantum_rows: self.config.selection_quantum_rows,
        };
        let local_pack = plan_rolling_group_index_packs(
            &row_group_offsets,
            &row_groups,
            self.group_indices.len(),
            local_config,
            1,
        )?
        .packs
        .into_iter()
        .next()
        .ok_or_else(|| rejected("rolling row pack accumulator produced no physical pack"))?;
        let row_indices = local_pack
            .iter()
            .map(|local_row| self.pending[*local_row].row_index)
            .collect::<Vec<_>>();
        ensure(
            row_indices.contains(&oldest_pending_row),
            "rolling row pack did not advance its oldest row",
        )?;
        let max_selected_row_offset = row_indices
            .iter()
            .map(|row_index| row_index - oldest_pending_row)
            .max()
            .unwrap_or(0);
        let mut selected = vec![false; self.pending.len()];
        for local_row in local_pack {
            selected[local_row] = true;
        }
        let old_pending = std::mem::take(&mut self.pending);
        self.pending = old_pending
            .into_iter()
            .zip(selected)
            .filter_map(|(row, selected)| (!selected).then_some(row))
            .collect();
        let emission = RollingExpertRowPackEmission {
            row_indices,
            emitted_pack_index: self.emitted_packs,
            admitted_rows: self.admitted_rows,
            oldest_pending_row,
            max_selected_row_offset,
            deadline_row_exclusive,
        };
        self.emitted_packs += 1;
        Ok(emission)
    }
}

/// Packs complete routed rows while guaranteeing progress on the oldest logical
/// scheduler chunk. Rows selected from later chunks are useful lookahead work.
pub fn plan_rolling_expert_row_packs(
    entries: &[CompletionRoutePlanEntry],
    row_count: usize,
    config: RollingExpertRowPackConfig,
) -> Result<RollingExpertRowPackPlan, GlmrtError> {
    plan_rolling_expert_row_packs_limited(entries, row_count, config, usize::MAX)
}

fn plan_rolling_expert_row_packs_limited(
    entries: &[CompletionRoutePlanEntry],
    row_count: usize,
    config: RollingExpertRowPackConfig,
    pack_limit: usize,
) -> Result<RollingExpertRowPackPlan, GlmrtError> {
    ensure(row_count > 0, "rolling row pack requires at least one row")?;
    ensure(pack_limit > 0, "rolling row pack limit must be non-zero")?;
    validate_rolling_expert_row_pack_config(config)?;

    let mut routes_per_row = vec![0_usize; row_count];
    for entry in entries {
        ensure(
            entry.row_index < row_count,
            format!(
                "rolling row pack route row {} exceeds row count {row_count}",
                entry.row_index
            ),
        )?;
        routes_per_row[entry.row_index] += 1;
    }
    let mut row_group_offsets = Vec::with_capacity(row_count + 1);
    row_group_offsets.push(0_usize);
    for route_count in &routes_per_row {
        let next_offset = row_group_offsets
            .last()
            .copied()
            .expect("rolling row offsets start at zero")
            .checked_add(*route_count)
            .ok_or_else(|| rejected("rolling row pack route offsets overflow"))?;
        row_group_offsets.push(next_offset);
    }
    let mut group_indices =
        HashMap::<(usize, usize), usize>::with_capacity(entries.len().min(1024));
    let mut row_group_cursors = row_group_offsets[..row_count].to_vec();
    let mut row_groups = vec![usize::MAX; entries.len()];
    for entry in entries {
        let group = (entry.expert_id, entry.intermediate_rows);
        let next_group_index = group_indices.len();
        let group_index = *group_indices.entry(group).or_insert(next_group_index);
        let row_start = row_group_offsets[entry.row_index];
        let row_cursor = row_group_cursors[entry.row_index];
        ensure(
            !row_groups[row_start..row_cursor].contains(&group_index),
            format!(
                "rolling row pack row {} repeats expert group {:?}",
                entry.row_index, group
            ),
        )?;
        row_groups[row_cursor] = group_index;
        row_group_cursors[entry.row_index] += 1;
    }
    for (row_index, route_count) in routes_per_row.iter().enumerate() {
        ensure(
            *route_count > 0,
            format!("rolling row pack row {row_index} has no routes"),
        )?;
    }
    plan_rolling_group_index_packs(
        &row_group_offsets,
        &row_groups,
        group_indices.len(),
        config,
        pack_limit,
    )
}

fn plan_rolling_group_index_packs(
    row_group_offsets: &[usize],
    row_groups: &[usize],
    group_count: usize,
    config: RollingExpertRowPackConfig,
    pack_limit: usize,
) -> Result<RollingExpertRowPackPlan, GlmrtError> {
    let row_count = row_group_offsets
        .len()
        .checked_sub(1)
        .ok_or_else(|| rejected("rolling row group offsets are empty"))?;
    ensure(row_count > 0, "rolling row pack requires at least one row")?;
    ensure(pack_limit > 0, "rolling row pack limit must be non-zero")?;
    ensure(
        row_group_offsets[0] == 0
            && row_group_offsets[row_count] == row_groups.len()
            && row_group_offsets.windows(2).all(|pair| pair[0] < pair[1]),
        "rolling row group offsets are not a complete non-empty partition",
    )?;
    validate_rolling_expert_row_pack_config(config)?;
    debug_assert!(row_groups.iter().all(|group| *group < group_count));

    let groups_for_row = |row_index: usize| {
        &row_groups[row_group_offsets[row_index]..row_group_offsets[row_index + 1]]
    };
    let mut rows_by_group = vec![Vec::<usize>::new(); group_count];
    for row_index in 0..row_count {
        for group_index in groups_for_row(row_index) {
            rows_by_group[*group_index].push(row_index);
        }
    }

    let mut remaining = vec![true; row_count];
    let mut remaining_rows = row_count;
    let mut packs = Vec::with_capacity(row_count.div_ceil(config.max_pack_rows));
    while remaining_rows > 0 && packs.len() < pack_limit {
        let oldest_row = remaining
            .iter()
            .position(|is_remaining| *is_remaining)
            .expect("remaining row count and bitmap agree");
        let oldest_chunk_end = ((oldest_row / config.logical_chunk_rows + 1)
            * config.logical_chunk_rows)
            .min(row_count);
        let mut pack = (oldest_row..oldest_chunk_end)
            .filter(|row_index| remaining[*row_index])
            .collect::<Vec<_>>();
        ensure(
            pack.len() <= config.max_pack_rows,
            "rolling row pack oldest logical chunk exceeds physical capacity",
        )?;

        let mut group_counts = vec![0_usize; group_count];
        for row_index in &pack {
            remaining[*row_index] = false;
            remaining_rows -= 1;
            for group_index in groups_for_row(*row_index) {
                group_counts[*group_index] += 1;
            }
        }
        let mut opened_tile_scores = vec![0_usize; row_count];
        let mut affinity_scores = vec![0_usize; row_count];
        for row_index in 0..row_count {
            if !remaining[row_index] {
                continue;
            }
            for group_index in groups_for_row(row_index) {
                let current_rows = group_counts[*group_index];
                opened_tile_scores[row_index] +=
                    usize::from(current_rows % config.expert_tile_rows == 0);
                affinity_scores[row_index] =
                    affinity_scores[row_index].saturating_add(current_rows);
            }
        }

        while pack.len() < config.max_pack_rows && remaining_rows > 0 {
            let next_oldest = remaining
                .iter()
                .position(|is_remaining| *is_remaining)
                .expect("remaining row count and bitmap agree");
            let candidate_end = next_oldest
                .saturating_add(config.lookahead_rows)
                .min(row_count);
            let selected_rows = config
                .selection_quantum_rows
                .min(config.max_pack_rows - pack.len())
                .min(remaining_rows);
            let mut best_candidates = Vec::with_capacity(candidate_end - next_oldest);
            for row_index in (next_oldest..candidate_end).filter(|row_index| remaining[*row_index])
            {
                best_candidates.push((
                    opened_tile_scores[row_index],
                    std::cmp::Reverse(affinity_scores[row_index]),
                    row_index,
                ));
            }
            ensure(
                !best_candidates.is_empty(),
                "rolling row pack lookahead did not expose a remaining row",
            )?;
            if best_candidates.len() > selected_rows {
                best_candidates.select_nth_unstable(selected_rows);
                best_candidates.truncate(selected_rows);
            }
            best_candidates.sort_unstable();
            let selected = best_candidates;
            let mut group_deltas = vec![0_usize; group_count];
            let mut changed_groups = Vec::with_capacity(selected.len() * 8);
            for (_, _, row_index) in &selected {
                for group_index in groups_for_row(*row_index) {
                    if group_deltas[*group_index] == 0 {
                        changed_groups.push(*group_index);
                    }
                    group_deltas[*group_index] += 1;
                }
            }
            for (_, _, row_index) in selected {
                remaining[row_index] = false;
                remaining_rows -= 1;
                pack.push(row_index);
            }
            for group_index in changed_groups {
                let old_count = group_counts[group_index];
                let delta = group_deltas[group_index];
                let new_count = old_count + delta;
                let old_opens_tile = old_count % config.expert_tile_rows == 0;
                let new_opens_tile = new_count % config.expert_tile_rows == 0;
                group_counts[group_index] = new_count;
                for row_index in &rows_by_group[group_index] {
                    if !remaining[*row_index] {
                        continue;
                    }
                    affinity_scores[*row_index] = affinity_scores[*row_index].saturating_add(delta);
                    match (old_opens_tile, new_opens_tile) {
                        (true, false) => opened_tile_scores[*row_index] -= 1,
                        (false, true) => opened_tile_scores[*row_index] += 1,
                        _ => {}
                    }
                }
            }
        }
        packs.push(pack);
    }

    Ok(RollingExpertRowPackPlan { packs })
}

fn validate_rolling_expert_row_pack_config(
    config: RollingExpertRowPackConfig,
) -> Result<(), GlmrtError> {
    ensure(
        config.logical_chunk_rows > 0,
        "rolling row pack logical chunk size must be non-zero",
    )?;
    ensure(
        config.max_pack_rows >= config.logical_chunk_rows,
        "rolling row pack capacity must cover one logical chunk",
    )?;
    ensure(
        config.lookahead_rows >= config.max_pack_rows,
        "rolling row pack lookahead must cover one physical pack",
    )?;
    ensure(
        config.expert_tile_rows > 0,
        "rolling row pack expert tile size must be non-zero",
    )?;
    ensure(
        config.selection_quantum_rows > 0,
        "rolling row pack selection quantum must be non-zero",
    )
}

pub fn plan_completion_first_routes(
    entries: &[CompletionRoutePlanEntry],
    row_count: usize,
    max_group_rows: usize,
) -> Result<CompletionFirstRoutePlan, GlmrtError> {
    ensure(max_group_rows > 0, "route group row cap must be non-zero")?;
    let mut remaining_by_row = vec![0_usize; row_count];
    let mut pending_by_expert = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (route_index, entry) in entries.iter().enumerate() {
        ensure(
            entry.row_index < row_count,
            format!(
                "completion route row {} exceeds row count {row_count}",
                entry.row_index
            ),
        )?;
        remaining_by_row[entry.row_index] = remaining_by_row[entry.row_index]
            .checked_add(1)
            .ok_or_else(|| rejected("completion route remaining count overflow"))?;
        pending_by_expert
            .entry((entry.expert_id, entry.intermediate_rows))
            .or_default()
            .push(route_index);
    }

    let mut groups = Vec::with_capacity(pending_by_expert.len());
    while !pending_by_expert.is_empty() {
        let minimum_remaining = remaining_by_row
            .iter()
            .copied()
            .filter(|remaining| *remaining > 0)
            .min()
            .ok_or_else(|| rejected("pending completion routes have no unfinished row"))?;
        let mut selected: Option<((usize, usize), usize, usize)> = None;
        for (key, route_indices) in &pending_by_expert {
            let minimum_bucket_rows = route_indices
                .iter()
                .filter(|route_index| {
                    remaining_by_row[entries[**route_index].row_index] == minimum_remaining
                })
                .count();
            if minimum_bucket_rows == 0 {
                continue;
            }
            let candidate = (*key, minimum_bucket_rows, route_indices.len());
            let replace = selected
                .map(|current| {
                    candidate.1 > current.1
                        || (candidate.1 == current.1 && candidate.2 > current.2)
                        || (candidate.1 == current.1
                            && candidate.2 == current.2
                            && candidate.0 < current.0)
                })
                .unwrap_or(true);
            if replace {
                selected = Some(candidate);
            }
        }
        let (selected_key, _, _) = selected.ok_or_else(|| {
            rejected("completion route planner could not select an expert for the minimum bucket")
        })?;
        let mut route_indices = pending_by_expert
            .remove(&selected_key)
            .expect("selected completion route expert exists");
        route_indices.sort_by_key(|route_index| {
            let entry = entries[*route_index];
            (
                remaining_by_row[entry.row_index],
                entry.row_index,
                *route_index,
            )
        });

        for chunk in route_indices.chunks(max_group_rows) {
            let mut completed_rows = Vec::new();
            for route_index in chunk {
                let row_index = entries[*route_index].row_index;
                remaining_by_row[row_index] = remaining_by_row[row_index]
                    .checked_sub(1)
                    .ok_or_else(|| rejected("completion route remaining count underflow"))?;
                if remaining_by_row[row_index] == 0 {
                    completed_rows.push(row_index);
                }
            }
            groups.push(CompletionFirstRouteGroup {
                route_indices: chunk.to_vec(),
                completed_rows,
                ready_after_rows: 0,
            });
        }
    }
    ensure(
        remaining_by_row.iter().all(|remaining| *remaining == 0),
        "completion route planner left unfinished rows",
    )?;

    let mut activation_row_order = Vec::with_capacity(row_count);
    let mut activation_staged = vec![false; row_count];
    for group in &mut groups {
        for route_index in &group.route_indices {
            let row_index = entries[*route_index].row_index;
            if !activation_staged[row_index] {
                activation_staged[row_index] = true;
                activation_row_order.push(row_index);
            }
        }
        group.ready_after_rows = activation_row_order.len();
    }

    Ok(CompletionFirstRoutePlan {
        groups,
        activation_row_order,
    })
}

fn ensure(condition: bool, reason: impl Into<String>) -> Result<(), GlmrtError> {
    if condition {
        Ok(())
    } else {
        Err(rejected(reason))
    }
}

fn rejected(reason: impl Into<String>) -> GlmrtError {
    GlmrtError::ExpertRoutePlanRejected {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prioritizes_completion_then_reuse_and_orders_activations_once() {
        let entries = vec![
            entry(0, 5),
            entry(0, 9),
            entry(1, 5),
            entry(2, 9),
            entry(2, 7),
            entry(3, 7),
        ];

        let plan = plan_completion_first_routes(&entries, 4, 256).unwrap();

        let experts = plan
            .groups
            .iter()
            .map(|group| entries[group.route_indices[0]].expert_id)
            .collect::<Vec<_>>();
        assert_eq!(experts, vec![5, 7, 9]);
        assert_eq!(plan.activation_row_order, vec![1, 0, 3, 2]);
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| group.ready_after_rows)
                .collect::<Vec<_>>(),
            vec![2, 4, 4]
        );
        assert_eq!(plan.groups[0].completed_rows, vec![1]);
        assert_eq!(plan.groups[1].completed_rows, vec![3]);
        assert_eq!(plan.groups[2].completed_rows, vec![0, 2]);
    }

    #[test]
    fn preserves_expert_reuse_across_the_group_cap() {
        let entries = (0..300).map(|row| entry(row, 11)).collect::<Vec<_>>();

        let plan = plan_completion_first_routes(&entries, 300, 256).unwrap();

        assert_eq!(plan.groups.len(), 2);
        assert_eq!(plan.groups[0].route_indices.len(), 256);
        assert_eq!(plan.groups[0].ready_after_rows, 256);
        assert_eq!(plan.groups[1].route_indices.len(), 44);
        assert_eq!(plan.groups[1].ready_after_rows, 300);
        assert_eq!(plan.activation_row_order, (0..300).collect::<Vec<_>>());
    }

    #[test]
    fn rolling_row_packs_complete_oldest_chunks_and_fill_from_lookahead() {
        let experts = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
        let entries = experts
            .into_iter()
            .enumerate()
            .map(|(row_index, expert_id)| CompletionRoutePlanEntry {
                row_index,
                expert_id,
                intermediate_rows: 1536,
            })
            .collect::<Vec<_>>();
        let config = RollingExpertRowPackConfig {
            logical_chunk_rows: 4,
            max_pack_rows: 8,
            lookahead_rows: 16,
            expert_tile_rows: 4,
            selection_quantum_rows: 2,
        };

        let plan = plan_rolling_expert_row_packs(&entries, experts.len(), config).unwrap();

        assert_eq!(plan.packs.len(), 2);
        assert!(plan.packs[0].starts_with(&[0, 1, 2, 3]));
        assert_eq!(&plan.packs[0][4..], &[8, 9, 10, 11]);
        assert!(plan
            .packs
            .iter()
            .all(|pack| !pack.is_empty() && pack.len() <= config.max_pack_rows));
        let mut emitted = plan.packs.iter().flatten().copied().collect::<Vec<_>>();
        emitted.sort_unstable();
        assert_eq!(emitted, (0..experts.len()).collect::<Vec<_>>());
    }

    #[test]
    fn rolling_row_packs_reject_rows_without_routes() {
        let error = plan_rolling_expert_row_packs(
            &[entry(0, 5)],
            2,
            RollingExpertRowPackConfig {
                logical_chunk_rows: 1,
                max_pack_rows: 2,
                lookahead_rows: 2,
                expert_tile_rows: 4,
                selection_quantum_rows: 1,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("row 1 has no routes"));
    }

    #[test]
    fn rolling_accumulator_waits_for_lookahead_and_advances_one_pack() {
        let config = streaming_config();
        let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
        for row_start in (0..4096).step_by(512) {
            let emissions = accumulator
                .push_chunk(&fixture_entries(row_start, 512), 512)
                .unwrap();
            if row_start < 3584 {
                assert!(emissions.is_empty());
            } else {
                assert_eq!(emissions.len(), 1);
                assert_eq!(emissions[0].row_indices.len(), 512);
                assert_eq!(emissions[0].admitted_rows, 4096);
                assert_eq!(emissions[0].oldest_pending_row, 0);
                assert!(emissions[0].max_selected_row_offset < 4096);
            }
        }
        assert_eq!(accumulator.pending_rows(), 3584);
        assert_eq!(accumulator.finish().unwrap().len(), 7);
        assert_eq!(accumulator.pending_rows(), 0);
    }

    #[test]
    fn rolling_accumulator_preserves_order_and_bounds_delay_at_required_sizes() {
        let config = streaming_config();
        for row_count in [255, 512, 1024, 2048, 4096, 8192, 16384] {
            let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
            let mut emissions = Vec::new();
            let mut emitted_online = vec![false; row_count];
            for row_start in (0..row_count).step_by(512) {
                let chunk_rows = (row_count - row_start).min(512);
                let pushed = accumulator
                    .push_chunk(&fixture_entries(row_start, chunk_rows), chunk_rows)
                    .unwrap();
                assert!(pushed.len() <= 1);
                for emission in &pushed {
                    for row_index in &emission.row_indices {
                        emitted_online[*row_index] = true;
                    }
                    let expected_deadline = emission
                        .admitted_rows
                        .checked_sub(config.lookahead_rows)
                        .filter(|deadline| *deadline > 0);
                    assert_eq!(emission.deadline_row_exclusive, expected_deadline);
                    if let Some(deadline) = expected_deadline {
                        assert!(emitted_online[..deadline].iter().all(|emitted| *emitted));
                    }
                }
                emissions.extend(pushed);
            }
            emissions.extend(accumulator.finish().unwrap());

            assert_eq!(
                emissions[0].admitted_rows,
                row_count.min(config.lookahead_rows)
            );
            assert_eq!(emissions.len(), row_count.div_ceil(config.max_pack_rows));
            assert!(emissions
                .iter()
                .take(emissions.len().saturating_sub(1))
                .all(|emission| emission.row_indices.len() == config.max_pack_rows));
            let final_pack_rows = match row_count % config.max_pack_rows {
                0 => config.max_pack_rows,
                rows => rows,
            };
            assert_eq!(emissions.last().unwrap().row_indices.len(), final_pack_rows);
            if row_count > config.lookahead_rows {
                assert!(emissions
                    .iter()
                    .any(|emission| emission.deadline_row_exclusive.is_some()));
            } else {
                assert!(emissions
                    .iter()
                    .all(|emission| emission.deadline_row_exclusive.is_none()));
            }

            let expected = (0..row_count)
                .map(|row| fixture_row_checksum(row))
                .collect::<Vec<_>>();
            let mut restored = vec![None; row_count];
            for emission in &emissions {
                assert!(emission.max_selected_row_offset < config.lookahead_rows);
                for row_index in &emission.row_indices {
                    assert!(restored[*row_index].is_none());
                    restored[*row_index] = Some(fixture_row_checksum(*row_index));
                }
            }
            assert_eq!(
                restored.into_iter().collect::<Option<Vec<_>>>(),
                Some(expected)
            );
        }
    }

    #[test]
    fn rolling_accumulator_dispatches_first_lookahead_by_two_windows_admitted() {
        let config = streaming_config();
        let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
        let mut emitted = vec![false; config.lookahead_rows * 2];

        for row_start in (0..emitted.len()).step_by(config.max_pack_rows) {
            let emissions = accumulator
                .push_chunk(
                    &fixture_entries(row_start, config.max_pack_rows),
                    config.max_pack_rows,
                )
                .unwrap();
            for emission in emissions {
                for row_index in emission.row_indices {
                    emitted[row_index] = true;
                }
            }
        }

        assert!(emitted[..config.lookahead_rows]
            .iter()
            .all(|emitted| *emitted));
    }

    #[test]
    fn rolling_accumulator_decouples_source_chunks_from_physical_packs() {
        let config = RollingExpertRowPackConfig {
            max_pack_rows: 256,
            ..streaming_config()
        };
        let total_rows = config.lookahead_rows * 2;
        let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
        let mut emitted = vec![false; total_rows];

        for row_start in (0..total_rows).step_by(512) {
            let emissions = accumulator
                .push_chunk(&fixture_entries(row_start, 512), 512)
                .unwrap();
            if row_start < config.lookahead_rows - 512 {
                assert!(emissions.is_empty());
            } else if row_start == config.lookahead_rows - 512 {
                assert_eq!(emissions.len(), 1);
            } else {
                assert_eq!(emissions.len(), 2);
            }
            for emission in emissions {
                assert_eq!(emission.row_indices.len(), config.max_pack_rows);
                for row_index in emission.row_indices {
                    emitted[row_index] = true;
                }
            }
        }

        assert!(emitted[..config.lookahead_rows]
            .iter()
            .all(|emitted| *emitted));
        let drained = accumulator.finish().unwrap();
        assert!(drained
            .iter()
            .all(|emission| emission.row_indices.len() <= config.max_pack_rows));
        for emission in drained {
            for row_index in emission.row_indices {
                emitted[row_index] = true;
            }
        }
        assert!(emitted.into_iter().all(|emitted| emitted));
    }

    #[test]
    fn rolling_accumulator_emits_when_a_short_tail_advances_the_deadline() {
        let config = RollingExpertRowPackConfig {
            max_pack_rows: 256,
            ..streaming_config()
        };
        let total_rows = config.lookahead_rows * 2 + 1;
        let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
        let mut emitted = vec![false; total_rows];

        for row_start in (0..total_rows).step_by(512) {
            let row_count = (total_rows - row_start).min(512);
            for emission in accumulator
                .push_chunk(&fixture_entries(row_start, row_count), row_count)
                .unwrap()
            {
                for row_index in emission.row_indices {
                    emitted[row_index] = true;
                }
            }
            let deadline = accumulator
                .admitted_rows()
                .saturating_sub(config.lookahead_rows);
            assert!(emitted[..deadline].iter().all(|emitted| *emitted));
        }
    }

    #[test]
    fn rolling_accumulator_rejects_admission_larger_than_lookahead() {
        let config = RollingExpertRowPackConfig {
            max_pack_rows: 256,
            ..streaming_config()
        };
        let mut accumulator = RollingExpertRowPackAccumulator::new(config).unwrap();
        let row_count = config.lookahead_rows + 1;

        assert!(accumulator
            .push_chunk(&fixture_entries(0, row_count), row_count)
            .unwrap_err()
            .to_string()
            .contains("lookahead capacity"));
    }

    #[test]
    fn rolling_accumulator_rejects_invalid_chunks_and_reuse() {
        let mut accumulator = RollingExpertRowPackAccumulator::new(streaming_config()).unwrap();
        let duplicate = vec![entry(0, 5), entry(0, 5)];
        assert!(accumulator
            .push_chunk(&duplicate, 1)
            .unwrap_err()
            .to_string()
            .contains("repeats expert group"));
        assert!(accumulator
            .push_chunk(&[entry(1, 5)], 1)
            .unwrap_err()
            .to_string()
            .contains("outside"));

        assert!(accumulator
            .push_chunk(&[entry(0, 5)], 1)
            .unwrap()
            .is_empty());
        assert_eq!(accumulator.finish().unwrap()[0].row_indices, vec![0]);
        assert!(accumulator
            .push_chunk(&[entry(1, 5)], 1)
            .unwrap_err()
            .to_string()
            .contains("finished"));
        assert!(accumulator
            .finish()
            .unwrap_err()
            .to_string()
            .contains("finished"));
    }

    fn streaming_config() -> RollingExpertRowPackConfig {
        RollingExpertRowPackConfig {
            logical_chunk_rows: 64,
            max_pack_rows: 512,
            lookahead_rows: 4096,
            expert_tile_rows: 32,
            selection_quantum_rows: 32,
        }
    }

    fn fixture_entries(row_start: usize, row_count: usize) -> Vec<CompletionRoutePlanEntry> {
        (row_start..row_start + row_count)
            .flat_map(|row_index| {
                (0..=row_index % 7).map(move |route| CompletionRoutePlanEntry {
                    row_index,
                    expert_id: (row_index * 17 + route * 29) % 64,
                    intermediate_rows: 1536,
                })
            })
            .collect()
    }

    fn fixture_row_checksum(row_index: usize) -> usize {
        (0..=row_index % 7)
            .map(|route| (row_index * 17 + route * 29) % 64)
            .sum()
    }

    fn entry(row_index: usize, expert_id: usize) -> CompletionRoutePlanEntry {
        CompletionRoutePlanEntry {
            row_index,
            expert_id,
            intermediate_rows: 1536,
        }
    }
}
