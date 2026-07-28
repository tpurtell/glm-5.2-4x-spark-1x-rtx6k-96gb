use anyhow::{bail, Context, Result};
use glmrt_core::CompletionFirstRoutePlan;

use super::{
    checked_u32, checked_usize, push_u16, push_u32, read_u16, read_u32, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor,
};

const STREAM_PLAN_MAGIC: &[u8; 8] = b"GLMSTRM2";
const STREAM_PLAN_VERSION: u16 = 1;
const STREAM_PLAN_HEADER_LEN: usize = 40;
const STREAM_PLAN_GROUP_LEN: usize = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertProtocolV2StreamRouteGroup {
    pub ready_after_rows: u32,
    pub route_indices: Vec<u32>,
    pub completed_rows: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertProtocolV2StreamPlan {
    pub row_count: u32,
    pub route_count: u32,
    pub activation_row_order: Vec<u32>,
    pub groups: Vec<ExpertProtocolV2StreamRouteGroup>,
}

impl ExpertProtocolV2StreamPlan {
    pub fn new(
        row_count: u32,
        route_count: u32,
        activation_row_order: Vec<u32>,
        groups: Vec<ExpertProtocolV2StreamRouteGroup>,
    ) -> Result<Self> {
        let plan = Self {
            row_count,
            route_count,
            activation_row_order,
            groups,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn from_completion_first(
        row_count: usize,
        route_count: usize,
        completion: &CompletionFirstRoutePlan,
    ) -> Result<Self> {
        let activation_row_order = completion
            .activation_row_order
            .iter()
            .map(|row| checked_u32(*row, "stream plan activation row"))
            .collect::<Result<Vec<_>>>()?;
        let groups = completion
            .groups
            .iter()
            .map(|group| {
                Ok(ExpertProtocolV2StreamRouteGroup {
                    ready_after_rows: checked_u32(
                        group.ready_after_rows,
                        "stream plan ready row count",
                    )?,
                    route_indices: group
                        .route_indices
                        .iter()
                        .map(|route| checked_u32(*route, "stream plan route index"))
                        .collect::<Result<Vec<_>>>()?,
                    completed_rows: group
                        .completed_rows
                        .iter()
                        .map(|row| checked_u32(*row, "stream plan completed row"))
                        .collect::<Result<Vec<_>>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(
            checked_u32(row_count, "stream plan row count")?,
            checked_u32(route_count, "stream plan route count")?,
            activation_row_order,
            groups,
        )
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let route_index_count = self.route_index_count()?;
        let completed_row_count = self.completed_row_count()?;
        let wire_bytes = self.encoded_len()?;
        let mut out = Vec::with_capacity(wire_bytes);
        out.extend_from_slice(STREAM_PLAN_MAGIC);
        push_u16(&mut out, STREAM_PLAN_VERSION);
        push_u16(&mut out, STREAM_PLAN_HEADER_LEN as u16);
        push_u32(&mut out, checked_u32(wire_bytes, "stream plan wire bytes")?);
        push_u32(&mut out, self.row_count);
        push_u32(&mut out, self.route_count);
        push_u32(
            &mut out,
            checked_u32(
                self.activation_row_order.len(),
                "stream plan activation row count",
            )?,
        );
        push_u32(
            &mut out,
            checked_u32(self.groups.len(), "stream plan group count")?,
        );
        push_u32(
            &mut out,
            checked_u32(route_index_count, "stream plan scheduled route count")?,
        );
        push_u32(
            &mut out,
            checked_u32(completed_row_count, "stream plan completed row count")?,
        );

        let mut route_offset = 0_usize;
        let mut completed_offset = 0_usize;
        for group in &self.groups {
            push_u32(&mut out, group.ready_after_rows);
            push_u32(
                &mut out,
                checked_u32(route_offset, "stream plan group route offset")?,
            );
            push_u32(
                &mut out,
                checked_u32(group.route_indices.len(), "stream plan group route count")?,
            );
            push_u32(
                &mut out,
                checked_u32(completed_offset, "stream plan group completion offset")?,
            );
            push_u32(
                &mut out,
                checked_u32(
                    group.completed_rows.len(),
                    "stream plan group completion count",
                )?,
            );
            route_offset += group.route_indices.len();
            completed_offset += group.completed_rows.len();
        }
        for row in &self.activation_row_order {
            push_u32(&mut out, *row);
        }
        for group in &self.groups {
            for route in &group.route_indices {
                push_u32(&mut out, *route);
            }
        }
        for group in &self.groups {
            for row in &group.completed_rows {
                push_u32(&mut out, *row);
            }
        }
        debug_assert_eq!(out.len(), wire_bytes);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < STREAM_PLAN_HEADER_LEN {
            bail!(
                "ExpertProtocolV2 stream plan frame too short: {}",
                bytes.len()
            );
        }
        if bytes.get(..STREAM_PLAN_MAGIC.len()) != Some(STREAM_PLAN_MAGIC) {
            bail!("ExpertProtocolV2 stream plan magic mismatch");
        }
        let version = read_u16(bytes, 8, "stream plan version")?;
        if version != STREAM_PLAN_VERSION {
            bail!("unsupported ExpertProtocolV2 stream plan version {version}");
        }
        let header_len = read_u16(bytes, 10, "stream plan header length")? as usize;
        if header_len != STREAM_PLAN_HEADER_LEN {
            bail!(
                "ExpertProtocolV2 stream plan header length {header_len} did not match {STREAM_PLAN_HEADER_LEN}"
            );
        }
        let wire_bytes = read_u32(bytes, 12, "stream plan wire bytes")? as usize;
        if wire_bytes != bytes.len() {
            bail!(
                "ExpertProtocolV2 stream plan wire bytes {wire_bytes} did not match {}",
                bytes.len()
            );
        }
        let row_count = read_u32(bytes, 16, "stream plan row count")?;
        let route_count = read_u32(bytes, 20, "stream plan route count")?;
        let activation_count = read_u32(bytes, 24, "stream plan activation count")? as usize;
        let group_count = read_u32(bytes, 28, "stream plan group count")? as usize;
        let route_index_count = read_u32(bytes, 32, "stream plan scheduled route count")? as usize;
        let completed_row_count = read_u32(bytes, 36, "stream plan completed row count")? as usize;
        let expected_bytes = stream_plan_encoded_len(
            activation_count,
            group_count,
            route_index_count,
            completed_row_count,
        )?;
        if expected_bytes != bytes.len() {
            bail!(
                "ExpertProtocolV2 stream plan section bytes {expected_bytes} did not match {}",
                bytes.len()
            );
        }

        let groups_start = STREAM_PLAN_HEADER_LEN;
        let activation_start = groups_start + group_count * STREAM_PLAN_GROUP_LEN;
        let routes_start = activation_start + activation_count * std::mem::size_of::<u32>();
        let completed_start = routes_start + route_index_count * std::mem::size_of::<u32>();
        let activation_row_order = read_u32_vector(
            bytes,
            activation_start,
            activation_count,
            "stream plan activation rows",
        )?;
        let route_indices = read_u32_vector(
            bytes,
            routes_start,
            route_index_count,
            "stream plan route indices",
        )?;
        let completed_rows = read_u32_vector(
            bytes,
            completed_start,
            completed_row_count,
            "stream plan completed rows",
        )?;
        let mut groups = Vec::with_capacity(group_count);
        for group_index in 0..group_count {
            let offset = groups_start + group_index * STREAM_PLAN_GROUP_LEN;
            let ready_after_rows = read_u32(bytes, offset, "stream group ready rows")?;
            let route_offset = read_u32(bytes, offset + 4, "stream group route offset")? as usize;
            let route_count = read_u32(bytes, offset + 8, "stream group route count")? as usize;
            let completion_offset =
                read_u32(bytes, offset + 12, "stream group completion offset")? as usize;
            let completion_count =
                read_u32(bytes, offset + 16, "stream group completion count")? as usize;
            let route_end = route_offset
                .checked_add(route_count)
                .context("ExpertProtocolV2 stream group route range overflow")?;
            let completion_end = completion_offset
                .checked_add(completion_count)
                .context("ExpertProtocolV2 stream group completion range overflow")?;
            groups.push(ExpertProtocolV2StreamRouteGroup {
                ready_after_rows,
                route_indices: route_indices
                    .get(route_offset..route_end)
                    .with_context(|| {
                        format!(
                            "ExpertProtocolV2 stream group {group_index} route range is invalid"
                        )
                    })?
                    .to_vec(),
                completed_rows: completed_rows
                    .get(completion_offset..completion_end)
                    .with_context(|| {
                        format!(
                            "ExpertProtocolV2 stream group {group_index} completion range is invalid"
                        )
                    })?
                    .to_vec(),
            });
        }
        Self::new(row_count, route_count, activation_row_order, groups)
    }

    pub fn validate_against_request(
        &self,
        rows: &[ExpertProtocolV2RowDescriptor],
        routes: &[ExpertProtocolV2RouteEntry],
    ) -> Result<()> {
        self.validate()?;
        anyhow::ensure!(
            rows.len() == self.row_count as usize,
            "ExpertProtocolV2 stream plan rows {} did not match request rows {}",
            self.row_count,
            rows.len()
        );
        anyhow::ensure!(
            routes.len() == self.route_count as usize,
            "ExpertProtocolV2 stream plan routes {} did not match request routes {}",
            self.route_count,
            routes.len()
        );

        let mut route_declared = vec![false; routes.len()];
        for (row_index, row) in rows.iter().enumerate() {
            let start = row.route_offset as usize;
            let end = start
                .checked_add(row.route_count as usize)
                .context("ExpertProtocolV2 stream request row route range overflow")?;
            for route_index in start..end {
                let route = routes.get(route_index).with_context(|| {
                    format!(
                        "ExpertProtocolV2 stream request row {row_index} route range is invalid"
                    )
                })?;
                anyhow::ensure!(
                    route.row_index as usize == row_index,
                    "ExpertProtocolV2 stream request route {route_index} row {} did not match {row_index}",
                    route.row_index
                );
                route_declared[route_index] = true;
            }
        }
        anyhow::ensure!(
            route_declared.iter().all(|declared| *declared),
            "ExpertProtocolV2 stream request row ranges do not cover every route"
        );

        let mut activation_position = vec![0_usize; rows.len()];
        for (position, row) in self.activation_row_order.iter().enumerate() {
            activation_position[*row as usize] = position;
        }
        let mut remaining_by_row = rows
            .iter()
            .map(|row| row.route_count as usize)
            .collect::<Vec<_>>();
        for (group_index, group) in self.groups.iter().enumerate() {
            let mut expert_id = None;
            let mut actual_completed = Vec::new();
            for route_index in &group.route_indices {
                let route = &routes[*route_index as usize];
                anyhow::ensure!(
                    activation_position[route.row_index as usize]
                        < group.ready_after_rows as usize,
                    "ExpertProtocolV2 stream group {group_index} route row {} is not staged by ready-after {}",
                    route.row_index,
                    group.ready_after_rows
                );
                if let Some(expected_expert) = expert_id {
                    anyhow::ensure!(
                        route.expert_id == expected_expert,
                        "ExpertProtocolV2 stream group {group_index} mixes experts {expected_expert} and {}",
                        route.expert_id
                    );
                } else {
                    expert_id = Some(route.expert_id);
                }
                let remaining = &mut remaining_by_row[route.row_index as usize];
                *remaining = remaining
                    .checked_sub(1)
                    .context("ExpertProtocolV2 stream row remaining-route underflow")?;
                if *remaining == 0 {
                    actual_completed.push(route.row_index);
                }
            }
            anyhow::ensure!(
                actual_completed == group.completed_rows,
                "ExpertProtocolV2 stream group {group_index} completed rows {:?}, expected {:?}",
                group.completed_rows,
                actual_completed
            );
        }
        anyhow::ensure!(
            remaining_by_row.iter().all(|remaining| *remaining == 0),
            "ExpertProtocolV2 stream plan left unfinished request rows"
        );
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.row_count > 0,
            "ExpertProtocolV2 stream plan must contain rows"
        );
        anyhow::ensure!(
            self.route_count > 0,
            "ExpertProtocolV2 stream plan must contain routes"
        );
        anyhow::ensure!(
            self.activation_row_order.len() == self.row_count as usize,
            "ExpertProtocolV2 stream plan activation rows {} did not match row count {}",
            self.activation_row_order.len(),
            self.row_count
        );
        validate_permutation(
            &self.activation_row_order,
            self.row_count as usize,
            "activation row",
        )?;
        let mut scheduled_routes = Vec::with_capacity(self.route_count as usize);
        let mut completed_rows = Vec::with_capacity(self.row_count as usize);
        let mut previous_ready = 0_u32;
        for (group_index, group) in self.groups.iter().enumerate() {
            anyhow::ensure!(
                !group.route_indices.is_empty(),
                "ExpertProtocolV2 stream group {group_index} has no routes"
            );
            anyhow::ensure!(
                group.ready_after_rows >= previous_ready
                    && group.ready_after_rows <= self.row_count,
                "ExpertProtocolV2 stream group {group_index} ready-after {} is outside {previous_ready}..={}",
                group.ready_after_rows,
                self.row_count
            );
            previous_ready = group.ready_after_rows;
            scheduled_routes.extend_from_slice(&group.route_indices);
            completed_rows.extend_from_slice(&group.completed_rows);
        }
        validate_permutation(
            &scheduled_routes,
            self.route_count as usize,
            "scheduled route",
        )?;
        validate_permutation(&completed_rows, self.row_count as usize, "completed row")?;
        self.encoded_len()?;
        Ok(())
    }

    fn route_index_count(&self) -> Result<usize> {
        self.groups.iter().try_fold(0_usize, |count, group| {
            count
                .checked_add(group.route_indices.len())
                .context("ExpertProtocolV2 stream scheduled route count overflow")
        })
    }

    fn completed_row_count(&self) -> Result<usize> {
        self.groups.iter().try_fold(0_usize, |count, group| {
            count
                .checked_add(group.completed_rows.len())
                .context("ExpertProtocolV2 stream completed row count overflow")
        })
    }

    fn encoded_len(&self) -> Result<usize> {
        stream_plan_encoded_len(
            self.activation_row_order.len(),
            self.groups.len(),
            self.route_index_count()?,
            self.completed_row_count()?,
        )
    }
}

fn stream_plan_encoded_len(
    activation_count: usize,
    group_count: usize,
    route_index_count: usize,
    completed_row_count: usize,
) -> Result<usize> {
    let group_bytes = group_count
        .checked_mul(STREAM_PLAN_GROUP_LEN)
        .context("ExpertProtocolV2 stream group byte count overflow")?;
    let vector_values = activation_count
        .checked_add(route_index_count)
        .and_then(|count| count.checked_add(completed_row_count))
        .context("ExpertProtocolV2 stream vector value count overflow")?;
    let vector_bytes = vector_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("ExpertProtocolV2 stream vector byte count overflow")?;
    STREAM_PLAN_HEADER_LEN
        .checked_add(group_bytes)
        .and_then(|bytes| bytes.checked_add(vector_bytes))
        .context("ExpertProtocolV2 stream plan byte count overflow")
}

fn read_u32_vector(bytes: &[u8], start: usize, count: usize, label: &str) -> Result<Vec<u32>> {
    (0..count)
        .map(|index| {
            let offset = start
                .checked_add(index * std::mem::size_of::<u32>())
                .context("ExpertProtocolV2 stream vector offset overflow")?;
            read_u32(bytes, offset, label)
        })
        .collect()
}

fn validate_permutation(values: &[u32], count: usize, label: &str) -> Result<()> {
    if values.len() != count {
        bail!(
            "ExpertProtocolV2 stream plan {label} count {} did not match {count}",
            values.len()
        );
    }
    let mut seen = vec![false; count];
    for value in values {
        let index = checked_usize(*value as u64, label)?;
        let slot = seen.get_mut(index).with_context(|| {
            format!("ExpertProtocolV2 stream plan {label} {index} exceeds count {count}")
        })?;
        anyhow::ensure!(
            !*slot,
            "ExpertProtocolV2 stream plan {label} {index} appears more than once"
        );
        *slot = true;
    }
    Ok(())
}
