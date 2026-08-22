use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub(super) const TARGET_KV_PAGE_TOKENS: usize = 64;

type NodeId = usize;

const ROOT_NODE_ID: NodeId = 0;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PageDelta {
    retain_parent_pages: usize,
    pages: Vec<u32>,
}

#[derive(Debug)]
struct RadixNode {
    parent: Option<NodeId>,
    edge_tokens: Vec<u32>,
    prefix_tokens: usize,
    page_delta: PageDelta,
    children: HashMap<u32, NodeId>,
    lock_ref: usize,
    last_access: u64,
}

impl RadixNode {
    fn root() -> Self {
        Self {
            parent: None,
            edge_tokens: Vec::new(),
            prefix_tokens: 0,
            page_delta: PageDelta::default(),
            children: HashMap::new(),
            lock_ref: 0,
            last_access: 0,
        }
    }
}

#[derive(Debug)]
struct TargetKvRadixInner {
    free_pages: BTreeSet<u32>,
    nodes: Vec<Option<RadixNode>>,
    free_node_ids: Vec<NodeId>,
    access_clock: u64,
    active_reservations: usize,
    reserved_private_pages: usize,
    cache_hit_requests: u64,
    cache_miss_requests: u64,
    cache_hit_tokens: u64,
    evicted_nodes: u64,
    evicted_pages: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TargetKvBoundaryCopy {
    pub(super) source_page: u32,
    pub(super) destination_page: u32,
    pub(super) valid_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TargetKvPublishedPrefix {
    pub(super) matched_existing_tokens: usize,
    pub(super) physical_pages: Vec<u32>,
    pub(super) published_pages: usize,
    pub(super) duplicate_pages_freed: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TargetKvRadixStats {
    pub(super) total_pages: usize,
    pub(super) free_pages: usize,
    pub(super) cached_pages: usize,
    pub(super) allocated_private_pages: usize,
    pub(super) pinned_cached_pages: usize,
    pub(super) reserved_private_pages: usize,
    pub(super) active_reservations: usize,
    pub(super) radix_nodes: usize,
    pub(super) cache_hit_requests: u64,
    pub(super) cache_miss_requests: u64,
    pub(super) cache_hit_tokens: u64,
    pub(super) evicted_nodes: u64,
    pub(super) evicted_pages: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TargetKvExactSubtreeEviction {
    Evicted { pages: usize },
    AlreadyAbsent { matched_tokens: usize },
}

#[derive(Debug)]
pub(super) struct TargetKvRadixManager {
    total_pages: usize,
    max_active_requests: usize,
    inner: Mutex<TargetKvRadixInner>,
}

impl TargetKvRadixManager {
    pub(super) fn new(total_tokens: usize, max_active_requests: usize) -> Result<Self> {
        anyhow::ensure!(
            total_tokens > 0 && total_tokens % TARGET_KV_PAGE_TOKENS == 0,
            "target KV pool tokens {total_tokens} must be a positive multiple of {TARGET_KV_PAGE_TOKENS}"
        );
        anyhow::ensure!(
            max_active_requests > 0,
            "target KV radix active-request limit is zero"
        );
        let total_pages = total_tokens / TARGET_KV_PAGE_TOKENS;
        anyhow::ensure!(
            total_pages <= u32::MAX as usize,
            "target KV pool has {total_pages} pages, exceeding u32 page IDs"
        );
        Ok(Self {
            total_pages,
            max_active_requests,
            inner: Mutex::new(TargetKvRadixInner {
                free_pages: (0..u32::try_from(total_pages).expect("validated page count"))
                    .collect(),
                nodes: vec![Some(RadixNode::root())],
                free_node_ids: Vec::new(),
                access_clock: 0,
                active_reservations: 0,
                reserved_private_pages: 0,
                cache_hit_requests: 0,
                cache_miss_requests: 0,
                cache_hit_tokens: 0,
                evicted_nodes: 0,
                evicted_pages: 0,
            }),
        })
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        prompt_token_ids: &[usize],
        logical_capacity_tokens: usize,
    ) -> Result<TargetKvRadixReservation> {
        anyhow::ensure!(
            logical_capacity_tokens > 0,
            "target KV request capacity is empty"
        );
        anyhow::ensure!(
            prompt_token_ids.len() <= logical_capacity_tokens,
            "prompt has {} tokens but target KV capacity is {logical_capacity_tokens}",
            prompt_token_ids.len()
        );
        let capacity_pages = pages_for_tokens(logical_capacity_tokens)?;
        anyhow::ensure!(
            capacity_pages <= self.total_pages,
            "target KV request needs {capacity_pages} pages but the pool has {}",
            self.total_pages
        );
        let prompt_tokens = token_ids_u32(prompt_token_ids)?;

        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking target KV radix manager failed: {error}"))?;
        anyhow::ensure!(
            inner.active_reservations < self.max_active_requests,
            "target KV active request limit exhausted: active={} max={}",
            inner.active_reservations,
            self.max_active_requests
        );

        let matched = match_prefix(&mut inner, &prompt_tokens)?;
        touch_path(&mut inner, matched.terminal);
        inc_lock_ref(&mut inner, matched.terminal)?;

        let will_append = logical_capacity_tokens > matched.tokens;
        let shared_direct_pages = if will_append && matched.tokens % TARGET_KV_PAGE_TOKENS != 0 {
            matched.tokens / TARGET_KV_PAGE_TOKENS
        } else {
            pages_for_tokens(matched.tokens)?
        };
        let private_credit_pages = capacity_pages
            .checked_sub(shared_direct_pages)
            .context("matched target KV pages exceed request capacity")?;
        let proposed_reserved_private = inner
            .reserved_private_pages
            .checked_add(private_credit_pages)
            .context("target KV private capacity credit overflow")?;
        let pinned_cached_pages = pinned_cached_pages(&inner);
        if proposed_reserved_private
            .checked_add(pinned_cached_pages)
            .context("target KV guaranteed-capacity accounting overflow")?
            > self.total_pages
        {
            dec_lock_ref(&mut inner, matched.terminal);
            anyhow::bail!(
                "target KV guaranteed capacity exhausted: requested_private_pages={} reserved_private_pages={} pinned_cached_pages={} total_pages={}",
                private_credit_pages,
                inner.reserved_private_pages,
                pinned_cached_pages,
                self.total_pages
            );
        }

        inner.reserved_private_pages = proposed_reserved_private;
        inner.active_reservations += 1;
        if matched.tokens > 0 {
            inner.cache_hit_requests = inner.cache_hit_requests.saturating_add(1);
            inner.cache_hit_tokens = inner.cache_hit_tokens.saturating_add(matched.tokens as u64);
        } else {
            inner.cache_miss_requests = inner.cache_miss_requests.saturating_add(1);
        }

        let mut page_table = matched.page_table[..shared_direct_pages].to_vec();
        let mut private_pages = Vec::new();
        let mut boundary_copy = None;
        let allocation = (|| -> Result<()> {
            if shared_direct_pages < matched.page_table.len() {
                let source_page = *matched
                    .page_table
                    .last()
                    .expect("a partial matched prefix owns a boundary page");
                let destination_page = allocate_physical_page(&mut inner)?;
                private_pages.push(destination_page);
                page_table.push(destination_page);
                boundary_copy = Some(TargetKvBoundaryCopy {
                    source_page,
                    destination_page,
                    valid_tokens: matched.tokens % TARGET_KV_PAGE_TOKENS,
                });
            }
            let prompt_pages = pages_for_tokens(prompt_tokens.len())?;
            while page_table.len() < prompt_pages {
                let page = allocate_physical_page(&mut inner)?;
                private_pages.push(page);
                page_table.push(page);
            }
            Ok(())
        })();
        if let Err(error) = allocation {
            inner.free_pages.extend(private_pages.iter().copied());
            inner.reserved_private_pages = inner
                .reserved_private_pages
                .checked_sub(private_credit_pages)
                .expect("target KV private credit rollback underflow");
            inner.active_reservations = inner
                .active_reservations
                .checked_sub(1)
                .expect("target KV active reservation rollback underflow");
            dec_lock_ref(&mut inner, matched.terminal);
            return Err(error);
        }
        debug_assert!(private_pages.len() <= private_credit_pages);
        debug_validate(&inner, self.total_pages);

        Ok(TargetKvRadixReservation {
            manager: Arc::clone(self),
            matched_terminal: matched.terminal,
            matched_prefix_tokens: matched.tokens,
            logical_capacity_tokens,
            private_credit_pages,
            page_table,
            private_pages,
            boundary_copy,
            active: true,
        })
    }

    pub(super) fn stats(&self) -> TargetKvRadixStats {
        let inner = self
            .inner
            .lock()
            .expect("target KV radix manager lock poisoned");
        let cached_pages = cached_pages(&inner);
        TargetKvRadixStats {
            total_pages: self.total_pages,
            free_pages: inner.free_pages.len(),
            cached_pages,
            allocated_private_pages: self
                .total_pages
                .saturating_sub(inner.free_pages.len())
                .saturating_sub(cached_pages),
            pinned_cached_pages: pinned_cached_pages(&inner),
            reserved_private_pages: inner.reserved_private_pages,
            active_reservations: inner.active_reservations,
            radix_nodes: inner.nodes.iter().flatten().count().saturating_sub(1),
            cache_hit_requests: inner.cache_hit_requests,
            cache_miss_requests: inner.cache_miss_requests,
            cache_hit_tokens: inner.cache_hit_tokens,
            evicted_nodes: inner.evicted_nodes,
            evicted_pages: inner.evicted_pages,
        }
    }

    /// Removes one exact, inactive leaf without disturbing any shorter cached
    /// prefix. Startup uses this to discard a synthetic long-context seed once
    /// its graph-capture consumers have finished.
    pub(super) fn evict_exact_inactive_leaf(&self, token_ids: &[usize]) -> Result<usize> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "cannot evict the target KV radix root"
        );
        let token_ids = token_ids_u32(token_ids)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking target KV radix manager failed: {error}"))?;
        let matched = match_prefix(&mut inner, &token_ids)?;
        anyhow::ensure!(
            matched.tokens == token_ids.len(),
            "target KV radix exact-prefix eviction matched {} of {} tokens",
            matched.tokens,
            token_ids.len()
        );
        let evicted_pages = evict_leaf(&mut inner, matched.terminal)?;
        debug_validate(&inner, self.total_pages);
        Ok(evicted_pages)
    }

    /// Removes an exact inactive prefix and every cached descendant. Startup
    /// uses this for synthetic capture trees whose surviving branches depend
    /// on how much physical KV capacity was available during prewarm.
    pub(super) fn evict_exact_inactive_subtree_if_present(
        &self,
        token_ids: &[usize],
    ) -> Result<TargetKvExactSubtreeEviction> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "cannot evict the target KV radix root"
        );
        let token_ids = token_ids_u32(token_ids)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking target KV radix manager failed: {error}"))?;
        let matched = match_prefix(&mut inner, &token_ids)?;
        if matched.tokens != token_ids.len() {
            debug_validate(&inner, self.total_pages);
            return Ok(TargetKvExactSubtreeEviction::AlreadyAbsent {
                matched_tokens: matched.tokens,
            });
        }
        let evicted_pages = evict_subtree(&mut inner, matched.terminal)?;
        debug_validate(&inner, self.total_pages);
        Ok(TargetKvExactSubtreeEviction::Evicted {
            pages: evicted_pages,
        })
    }
}

struct PrefixMatch {
    terminal: NodeId,
    tokens: usize,
    page_table: Vec<u32>,
}

#[derive(Debug)]
pub(super) struct TargetKvRadixReservation {
    manager: Arc<TargetKvRadixManager>,
    matched_terminal: NodeId,
    matched_prefix_tokens: usize,
    logical_capacity_tokens: usize,
    private_credit_pages: usize,
    page_table: Vec<u32>,
    private_pages: Vec<u32>,
    boundary_copy: Option<TargetKvBoundaryCopy>,
    active: bool,
}

impl TargetKvRadixReservation {
    pub(super) fn matched_prefix_tokens(&self) -> usize {
        self.matched_prefix_tokens
    }

    pub(super) fn physical_pages(&self) -> &[u32] {
        &self.page_table
    }

    pub(super) fn boundary_copy(&self) -> Option<TargetKvBoundaryCopy> {
        self.boundary_copy
    }

    pub(super) fn take_boundary_copy(&mut self) -> Option<TargetKvBoundaryCopy> {
        self.boundary_copy.take()
    }

    pub(super) fn logical_capacity_tokens(&self) -> usize {
        self.logical_capacity_tokens
    }

    pub(super) fn reserved_private_pages(&self) -> usize {
        self.private_credit_pages
    }

    pub(super) fn ensure_materialized_through(&mut self, tokens: usize) -> Result<&[u32]> {
        anyhow::ensure!(
            tokens <= self.logical_capacity_tokens,
            "target KV materialization frontier {tokens} exceeds request capacity {}",
            self.logical_capacity_tokens
        );
        let required_pages = pages_for_tokens(tokens)?;
        let mut inner =
            self.manager.inner.lock().map_err(|error| {
                anyhow::anyhow!("locking target KV radix manager failed: {error}")
            })?;
        while self.page_table.len() < required_pages {
            anyhow::ensure!(
                self.private_pages.len() < self.private_credit_pages,
                "target KV request exhausted its private capacity credits"
            );
            let page = allocate_physical_page(&mut inner)?;
            self.private_pages.push(page);
            self.page_table.push(page);
        }
        debug_validate(&inner, self.manager.total_pages);
        Ok(&self.page_table)
    }

    pub(super) fn commit_prefix(
        mut self,
        committed_token_ids: &[usize],
        committed_tokens: usize,
    ) -> Result<TargetKvPublishedPrefix> {
        anyhow::ensure!(
            committed_tokens <= committed_token_ids.len(),
            "committed target KV frontier {committed_tokens} exceeds {} token IDs",
            committed_token_ids.len()
        );
        anyhow::ensure!(
            committed_tokens <= self.logical_capacity_tokens,
            "committed target KV frontier {committed_tokens} exceeds request capacity {}",
            self.logical_capacity_tokens
        );
        anyhow::ensure!(
            committed_tokens >= self.matched_prefix_tokens,
            "committed target KV frontier {committed_tokens} precedes matched prefix {}",
            self.matched_prefix_tokens
        );
        let committed_pages = pages_for_tokens(committed_tokens)?;
        anyhow::ensure!(
            committed_pages <= self.page_table.len(),
            "committed target KV frontier needs {committed_pages} pages but only {} are materialized",
            self.page_table.len()
        );
        let committed_ids = token_ids_u32(&committed_token_ids[..committed_tokens])?;

        let mut inner =
            self.manager.inner.lock().map_err(|error| {
                anyhow::anyhow!("locking target KV radix manager failed: {error}")
            })?;
        let current = match_prefix(&mut inner, &committed_ids)?;
        touch_path(&mut inner, current.terminal);

        let mut published_pages = 0;
        let canonical_pages = if current.tokens == committed_tokens {
            current.page_table
        } else {
            let reusable_full_pages = current.tokens / TARGET_KV_PAGE_TOKENS;
            anyhow::ensure!(
                reusable_full_pages <= current.page_table.len()
                    && reusable_full_pages <= committed_pages,
                "target KV concurrent-prefix page accounting is inconsistent"
            );
            let mut pages = current.page_table[..reusable_full_pages].to_vec();
            pages.extend_from_slice(&self.page_table[reusable_full_pages..committed_pages]);
            let suffix_tokens = committed_ids[current.tokens..].to_vec();
            let node_id = insert_radix_child(&mut inner, current.terminal, suffix_tokens, &pages)?;
            published_pages = inner.nodes[node_id]
                .as_ref()
                .expect("new target KV radix node is live")
                .page_delta
                .pages
                .len();
            pages
        };

        let retained_pages = canonical_pages.iter().copied().collect::<HashSet<_>>();
        let mut duplicate_pages_freed = 0;
        for page in &self.private_pages {
            if !retained_pages.contains(page) {
                let inserted = inner.free_pages.insert(*page);
                debug_assert!(inserted);
                duplicate_pages_freed += 1;
            }
        }
        finish_reservation(&mut inner, self.matched_terminal, self.private_credit_pages);
        self.active = false;
        debug_validate(&inner, self.manager.total_pages);

        Ok(TargetKvPublishedPrefix {
            matched_existing_tokens: current.tokens,
            physical_pages: canonical_pages,
            published_pages,
            duplicate_pages_freed,
        })
    }

    fn cancel(&mut self) {
        if !self.active {
            return;
        }
        let mut inner = self
            .manager
            .inner
            .lock()
            .expect("target KV radix manager lock poisoned during cancellation");
        inner.free_pages.extend(self.private_pages.iter().copied());
        finish_reservation(&mut inner, self.matched_terminal, self.private_credit_pages);
        self.active = false;
        debug_validate(&inner, self.manager.total_pages);
    }
}

impl Drop for TargetKvRadixReservation {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn token_ids_u32(token_ids: &[usize]) -> Result<Vec<u32>> {
    token_ids
        .iter()
        .copied()
        .map(|token_id| {
            u32::try_from(token_id)
                .with_context(|| format!("token ID {token_id} exceeds radix key width"))
        })
        .collect()
}

fn pages_for_tokens(tokens: usize) -> Result<usize> {
    if tokens == 0 {
        return Ok(0);
    }
    tokens
        .checked_add(TARGET_KV_PAGE_TOKENS - 1)
        .context("target KV page rounding overflow")
        .map(|rounded| rounded / TARGET_KV_PAGE_TOKENS)
}

fn next_access(inner: &mut TargetKvRadixInner) -> u64 {
    inner.access_clock = inner.access_clock.wrapping_add(1);
    inner.access_clock
}

fn match_prefix(inner: &mut TargetKvRadixInner, token_ids: &[u32]) -> Result<PrefixMatch> {
    let mut node_id = ROOT_NODE_ID;
    let mut matched_tokens = 0;
    while matched_tokens < token_ids.len() {
        let next_token = token_ids[matched_tokens];
        let child = inner.nodes[node_id]
            .as_ref()
            .expect("target KV radix parent is live")
            .children
            .get(&next_token)
            .copied();
        let Some(child_id) = child else {
            break;
        };
        let edge = &inner.nodes[child_id]
            .as_ref()
            .expect("target KV radix child is live")
            .edge_tokens;
        let remaining = &token_ids[matched_tokens..];
        let common = edge
            .iter()
            .zip(remaining)
            .take_while(|(left, right)| left == right)
            .count();
        debug_assert!(common > 0);
        if common < edge.len() {
            let split_id = split_node(inner, child_id, common)?;
            matched_tokens += common;
            node_id = split_id;
            break;
        }
        matched_tokens += common;
        node_id = child_id;
    }
    let page_table = materialize_page_table(inner, node_id)?;
    debug_assert_eq!(
        page_table.len(),
        pages_for_tokens(matched_tokens).expect("matched target KV token count is valid")
    );
    Ok(PrefixMatch {
        terminal: node_id,
        tokens: matched_tokens,
        page_table,
    })
}

fn split_node(
    inner: &mut TargetKvRadixInner,
    child_id: NodeId,
    split_edge_tokens: usize,
) -> Result<NodeId> {
    let child = inner.nodes[child_id]
        .as_ref()
        .context("target KV radix split child is absent")?;
    anyhow::ensure!(
        split_edge_tokens > 0 && split_edge_tokens < child.edge_tokens.len(),
        "invalid target KV radix split {split_edge_tokens}/{}",
        child.edge_tokens.len()
    );
    let parent_id = child.parent.context("cannot split target KV radix root")?;
    let parent_prefix_tokens = inner.nodes[parent_id]
        .as_ref()
        .context("target KV radix split parent is absent")?
        .prefix_tokens;
    let parent_pages = materialize_page_table(inner, parent_id)?;
    let child_pages = materialize_page_table(inner, child_id)?;
    let old_owned_pages = child.page_delta.pages.clone();
    let old_edge = child.edge_tokens.clone();
    let child_lock_ref = child.lock_ref;
    let child_last_access = child.last_access;
    let split_prefix_tokens = parent_prefix_tokens
        .checked_add(split_edge_tokens)
        .context("target KV radix split prefix overflow")?;
    let split_page_count = pages_for_tokens(split_prefix_tokens)?;
    let split_pages = child_pages[..split_page_count].to_vec();
    let split_delta = page_delta(&parent_pages, &split_pages);
    let child_delta = page_delta(&split_pages, &child_pages);

    let split_id = insert_node(
        inner,
        RadixNode {
            parent: Some(parent_id),
            edge_tokens: old_edge[..split_edge_tokens].to_vec(),
            prefix_tokens: split_prefix_tokens,
            page_delta: split_delta,
            children: HashMap::from([(old_edge[split_edge_tokens], child_id)]),
            lock_ref: child_lock_ref,
            last_access: child_last_access,
        },
    );
    {
        let child = inner.nodes[child_id]
            .as_mut()
            .expect("target KV radix split child remains live");
        child.parent = Some(split_id);
        child.edge_tokens = old_edge[split_edge_tokens..].to_vec();
        child.page_delta = child_delta;
    }
    inner.nodes[parent_id]
        .as_mut()
        .expect("target KV radix split parent remains live")
        .children
        .insert(old_edge[0], split_id);

    let mut redistributed_pages = inner.nodes[split_id]
        .as_ref()
        .expect("target KV radix split node is live")
        .page_delta
        .pages
        .clone();
    redistributed_pages.extend(
        inner.nodes[child_id]
            .as_ref()
            .expect("target KV radix split child is live")
            .page_delta
            .pages
            .iter()
            .copied(),
    );
    let mut old_owned_pages = old_owned_pages;
    old_owned_pages.sort_unstable();
    redistributed_pages.sort_unstable();
    debug_assert_eq!(old_owned_pages, redistributed_pages);
    Ok(split_id)
}

fn insert_radix_child(
    inner: &mut TargetKvRadixInner,
    parent_id: NodeId,
    edge_tokens: Vec<u32>,
    page_table: &[u32],
) -> Result<NodeId> {
    anyhow::ensure!(
        !edge_tokens.is_empty(),
        "target KV radix child edge is empty"
    );
    let parent = inner.nodes[parent_id]
        .as_ref()
        .context("target KV radix insertion parent is absent")?;
    anyhow::ensure!(
        !parent.children.contains_key(&edge_tokens[0]),
        "target KV radix insertion did not consume an existing matching child"
    );
    let prefix_tokens = parent
        .prefix_tokens
        .checked_add(edge_tokens.len())
        .context("target KV radix insertion prefix overflow")?;
    anyhow::ensure!(
        page_table.len() == pages_for_tokens(prefix_tokens)?,
        "target KV radix insertion page table has {} pages for {prefix_tokens} tokens",
        page_table.len()
    );
    let parent_pages = materialize_page_table(inner, parent_id)?;
    let access = next_access(inner);
    let child_key = edge_tokens[0];
    let node_id = insert_node(
        inner,
        RadixNode {
            parent: Some(parent_id),
            edge_tokens,
            prefix_tokens,
            page_delta: page_delta(&parent_pages, page_table),
            children: HashMap::new(),
            lock_ref: 0,
            last_access: access,
        },
    );
    inner.nodes[parent_id]
        .as_mut()
        .expect("target KV radix insertion parent remains live")
        .children
        .insert(child_key, node_id);
    Ok(node_id)
}

fn page_delta(parent_pages: &[u32], child_pages: &[u32]) -> PageDelta {
    let retain_parent_pages = parent_pages
        .iter()
        .zip(child_pages)
        .take_while(|(parent, child)| parent == child)
        .count();
    PageDelta {
        retain_parent_pages,
        pages: child_pages[retain_parent_pages..].to_vec(),
    }
}

fn materialize_page_table(inner: &TargetKvRadixInner, terminal: NodeId) -> Result<Vec<u32>> {
    let mut path = Vec::new();
    let mut node_id = terminal;
    while node_id != ROOT_NODE_ID {
        let node = inner.nodes[node_id]
            .as_ref()
            .context("target KV radix path contains an absent node")?;
        path.push(node_id);
        node_id = node
            .parent
            .context("non-root target KV radix node has no parent")?;
    }
    path.reverse();
    let mut pages = Vec::new();
    for node_id in path {
        let node = inner.nodes[node_id]
            .as_ref()
            .expect("materialized target KV radix node is live");
        anyhow::ensure!(
            node.page_delta.retain_parent_pages <= pages.len(),
            "target KV radix page delta retains {} of {} parent pages",
            node.page_delta.retain_parent_pages,
            pages.len()
        );
        pages.truncate(node.page_delta.retain_parent_pages);
        pages.extend_from_slice(&node.page_delta.pages);
    }
    let prefix_tokens = inner.nodes[terminal]
        .as_ref()
        .context("target KV radix terminal is absent")?
        .prefix_tokens;
    anyhow::ensure!(
        pages.len() == pages_for_tokens(prefix_tokens)?,
        "target KV radix terminal has {} pages for {prefix_tokens} tokens",
        pages.len()
    );
    Ok(pages)
}

fn touch_path(inner: &mut TargetKvRadixInner, terminal: NodeId) {
    let access = next_access(inner);
    let mut node_id = terminal;
    while node_id != ROOT_NODE_ID {
        let node = inner.nodes[node_id]
            .as_mut()
            .expect("touched target KV radix node is live");
        node.last_access = access;
        node_id = node
            .parent
            .expect("non-root target KV radix node has a parent");
    }
}

fn inc_lock_ref(inner: &mut TargetKvRadixInner, terminal: NodeId) -> Result<()> {
    let mut node_id = terminal;
    while node_id != ROOT_NODE_ID {
        let node = inner.nodes[node_id]
            .as_mut()
            .context("locking target KV radix path found an absent node")?;
        node.lock_ref = node
            .lock_ref
            .checked_add(1)
            .context("target KV radix lock reference overflow")?;
        node_id = node
            .parent
            .context("non-root target KV radix node has no parent")?;
    }
    Ok(())
}

fn dec_lock_ref(inner: &mut TargetKvRadixInner, terminal: NodeId) {
    let mut node_id = terminal;
    while node_id != ROOT_NODE_ID {
        let node = inner.nodes[node_id]
            .as_mut()
            .expect("unlocking target KV radix path found an absent node");
        node.lock_ref = node
            .lock_ref
            .checked_sub(1)
            .expect("target KV radix lock reference underflow");
        node_id = node
            .parent
            .expect("non-root target KV radix node has a parent");
    }
}

fn finish_reservation(
    inner: &mut TargetKvRadixInner,
    matched_terminal: NodeId,
    private_credit_pages: usize,
) {
    dec_lock_ref(inner, matched_terminal);
    inner.reserved_private_pages = inner
        .reserved_private_pages
        .checked_sub(private_credit_pages)
        .expect("target KV private capacity credit underflow");
    inner.active_reservations = inner
        .active_reservations
        .checked_sub(1)
        .expect("target KV active reservation accounting underflow");
}

fn allocate_physical_page(inner: &mut TargetKvRadixInner) -> Result<u32> {
    while inner.free_pages.is_empty() {
        evict_one_leaf(inner)
            .context("target KV page pool is exhausted after evicting every inactive radix leaf")?;
    }
    let page = *inner
        .free_pages
        .iter()
        .next()
        .expect("target KV free page set is nonempty");
    let removed = inner.free_pages.remove(&page);
    debug_assert!(removed);
    Ok(page)
}

fn evict_one_leaf(inner: &mut TargetKvRadixInner) -> Result<()> {
    let candidate = inner
        .nodes
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(node_id, node)| node.as_ref().map(|node| (node_id, node)))
        .filter(|(_, node)| node.lock_ref == 0 && node.children.is_empty())
        .min_by_key(|(node_id, node)| (node.last_access, *node_id))
        .map(|(node_id, _)| node_id)
        .context("no inactive target KV radix leaf is evictable")?;
    evict_leaf(inner, candidate).map(|_| ())
}

fn evict_leaf(inner: &mut TargetKvRadixInner, candidate: NodeId) -> Result<usize> {
    let candidate_node = inner.nodes.get(candidate).and_then(Option::as_ref);
    let candidate_node = candidate_node.context("target KV radix eviction node is absent")?;
    anyhow::ensure!(
        candidate != ROOT_NODE_ID,
        "cannot evict the target KV radix root"
    );
    anyhow::ensure!(
        candidate_node.lock_ref == 0,
        "target KV radix eviction leaf is active"
    );
    anyhow::ensure!(
        candidate_node.children.is_empty(),
        "target KV radix eviction prefix is not a leaf"
    );
    let node = inner.nodes[candidate]
        .take()
        .expect("selected target KV radix leaf is live");
    let parent_id = node
        .parent
        .expect("selected target KV radix leaf is not the root");
    let removed = inner.nodes[parent_id]
        .as_mut()
        .expect("target KV radix eviction parent is live")
        .children
        .remove(&node.edge_tokens[0]);
    debug_assert_eq!(removed, Some(candidate));
    for page in &node.page_delta.pages {
        let inserted = inner.free_pages.insert(*page);
        debug_assert!(inserted);
    }
    inner.evicted_nodes = inner.evicted_nodes.saturating_add(1);
    inner.evicted_pages = inner
        .evicted_pages
        .saturating_add(node.page_delta.pages.len() as u64);
    inner.free_node_ids.push(candidate);
    Ok(node.page_delta.pages.len())
}

fn evict_subtree(inner: &mut TargetKvRadixInner, candidate: NodeId) -> Result<usize> {
    anyhow::ensure!(
        candidate != ROOT_NODE_ID,
        "cannot evict the target KV radix root"
    );
    let mut traversal = vec![candidate];
    let mut parent_first = Vec::new();
    while let Some(node_id) = traversal.pop() {
        let node = inner
            .nodes
            .get(node_id)
            .and_then(Option::as_ref)
            .context("target KV radix subtree node is absent")?;
        anyhow::ensure!(
            node.lock_ref == 0,
            "target KV radix eviction subtree contains an active prefix"
        );
        parent_first.push(node_id);
        traversal.extend(node.children.values().copied());
    }

    let mut evicted_pages = 0_usize;
    for node_id in parent_first.into_iter().rev() {
        evicted_pages = evicted_pages
            .checked_add(evict_leaf(inner, node_id)?)
            .context("target KV radix subtree eviction page count overflow")?;
    }
    Ok(evicted_pages)
}

fn insert_node(inner: &mut TargetKvRadixInner, node: RadixNode) -> NodeId {
    if let Some(node_id) = inner.free_node_ids.pop() {
        debug_assert!(inner.nodes[node_id].is_none());
        inner.nodes[node_id] = Some(node);
        node_id
    } else {
        let node_id = inner.nodes.len();
        inner.nodes.push(Some(node));
        node_id
    }
}

fn cached_pages(inner: &TargetKvRadixInner) -> usize {
    inner
        .nodes
        .iter()
        .flatten()
        .map(|node| node.page_delta.pages.len())
        .sum()
}

fn pinned_cached_pages(inner: &TargetKvRadixInner) -> usize {
    inner
        .nodes
        .iter()
        .flatten()
        .filter(|node| node.lock_ref > 0)
        .map(|node| node.page_delta.pages.len())
        .sum()
}

fn debug_validate(inner: &TargetKvRadixInner, total_pages: usize) {
    #[cfg(debug_assertions)]
    {
        let cached = cached_pages(inner);
        let allocated_private = total_pages
            .checked_sub(inner.free_pages.len() + cached)
            .expect("target KV physical page accounting exceeds the pool");
        debug_assert!(allocated_private <= inner.reserved_private_pages);
        debug_assert!(pinned_cached_pages(inner) + inner.reserved_private_pages <= total_pages);
    }
}

#[cfg(test)]
mod tests {
    use super::{TargetKvExactSubtreeEviction, TargetKvRadixManager, TARGET_KV_PAGE_TOKENS};
    use std::sync::Arc;

    fn tokens(count: usize, seed: usize) -> Vec<usize> {
        (0..count).map(|index| seed * 100_000 + index).collect()
    }

    fn cache(manager: &Arc<TargetKvRadixManager>, token_ids: &[usize]) -> Vec<u32> {
        manager
            .reserve(token_ids, token_ids.len())
            .unwrap()
            .commit_prefix(token_ids, token_ids.len())
            .unwrap()
            .physical_pages
    }

    #[test]
    fn exact_token_branch_copies_only_the_partial_physical_page() {
        let manager = Arc::new(TargetKvRadixManager::new(8 * 64, 16).unwrap());
        let original = tokens(100, 1);
        let original_pages = cache(&manager, &original);

        let mut branch = original[..70].to_vec();
        branch.extend(tokens(30, 9));
        let reservation = manager.reserve(&branch, branch.len()).unwrap();
        assert_eq!(reservation.matched_prefix_tokens(), 70);
        assert_eq!(reservation.physical_pages()[0], original_pages[0]);
        assert_ne!(reservation.physical_pages()[1], original_pages[1]);
        let copy = reservation.boundary_copy().unwrap();
        assert_eq!(copy.source_page, original_pages[1]);
        assert_eq!(copy.destination_page, reservation.physical_pages()[1]);
        assert_eq!(copy.valid_tokens, 6);
        reservation.commit_prefix(&branch, branch.len()).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.cached_pages, 3);
        assert_eq!(stats.allocated_private_pages, 0);
        assert_eq!(stats.pinned_cached_pages, 0);
        assert_eq!(stats.radix_nodes, 3);
    }

    #[test]
    fn terminal_partial_hit_without_an_append_shares_the_page_directly() {
        let manager = Arc::new(TargetKvRadixManager::new(4 * 64, 16).unwrap());
        let original = tokens(70, 1);
        let pages = cache(&manager, &original);
        let exact = manager.reserve(&original, original.len()).unwrap();
        assert_eq!(exact.matched_prefix_tokens(), original.len());
        assert_eq!(exact.physical_pages(), pages.as_slice());
        assert_eq!(exact.boundary_copy(), None);
        assert_eq!(exact.reserved_private_pages(), 0);
    }

    #[test]
    fn capacity_credits_do_not_eagerly_materialize_output_pages() {
        let manager = Arc::new(TargetKvRadixManager::new(10 * 64, 16).unwrap());
        let prompt = tokens(64, 1);
        let reservation = manager.reserve(&prompt, 8 * 64).unwrap();
        assert_eq!(reservation.physical_pages().len(), 1);
        assert_eq!(reservation.reserved_private_pages(), 8);
        let stats = manager.stats();
        assert_eq!(stats.allocated_private_pages, 1);
        assert_eq!(stats.reserved_private_pages, 8);

        assert!(manager.reserve(&tokens(64, 2), 3 * 64).is_err());
        drop(reservation);
        assert!(manager.reserve(&tokens(64, 2), 3 * 64).is_ok());
    }

    #[test]
    fn materialization_is_incremental_and_commit_releases_the_unused_tail() {
        let manager = Arc::new(TargetKvRadixManager::new(8 * 64, 16).unwrap());
        let prompt = tokens(64, 1);
        let mut reservation = manager.reserve(&prompt, 4 * 64).unwrap();
        assert_eq!(
            reservation.ensure_materialized_through(193).unwrap().len(),
            4
        );
        let committed = tokens(130, 1);
        let published = reservation.commit_prefix(&committed, 130).unwrap();
        assert_eq!(published.physical_pages.len(), 3);
        assert_eq!(published.published_pages, 3);
        assert_eq!(published.duplicate_pages_freed, 1);
        let stats = manager.stats();
        assert_eq!(stats.cached_pages, 3);
        assert_eq!(stats.free_pages, 5);
        assert_eq!(stats.reserved_private_pages, 0);
    }

    #[test]
    fn concurrent_identical_misses_deduplicate_on_publish() {
        let manager = Arc::new(TargetKvRadixManager::new(6 * 64, 16).unwrap());
        let prompt = tokens(100, 1);
        let first = manager.reserve(&prompt, prompt.len()).unwrap();
        let second = manager.reserve(&prompt, prompt.len()).unwrap();
        assert_eq!(manager.stats().allocated_private_pages, 4);

        first.commit_prefix(&prompt, prompt.len()).unwrap();
        let second_publish = second.commit_prefix(&prompt, prompt.len()).unwrap();
        assert_eq!(second_publish.matched_existing_tokens, prompt.len());
        assert_eq!(second_publish.published_pages, 0);
        assert_eq!(second_publish.duplicate_pages_freed, 2);
        let stats = manager.stats();
        assert_eq!(stats.cached_pages, 2);
        assert_eq!(stats.free_pages, 4);
    }

    #[test]
    fn exact_leaf_eviction_preserves_its_cached_ancestor() {
        let manager = Arc::new(TargetKvRadixManager::new(4 * 64, 16).unwrap());
        let long = tokens(2 * TARGET_KV_PAGE_TOKENS, 1);
        cache(&manager, &long);

        let short = &long[..TARGET_KV_PAGE_TOKENS];
        let short_hit = manager.reserve(short, short.len()).unwrap();
        assert_eq!(short_hit.matched_prefix_tokens(), short.len());
        drop(short_hit);

        assert_eq!(manager.evict_exact_inactive_leaf(&long).unwrap(), 1);
        let stats = manager.stats();
        assert_eq!(stats.cached_pages, 1);
        assert_eq!(stats.evicted_nodes, 1);
        assert_eq!(stats.evicted_pages, 1);

        let short_hit = manager.reserve(short, short.len()).unwrap();
        assert_eq!(short_hit.matched_prefix_tokens(), short.len());
        drop(short_hit);
        let long_miss = manager.reserve(&long, long.len()).unwrap();
        assert_eq!(long_miss.matched_prefix_tokens(), short.len());
    }

    #[test]
    fn exact_leaf_eviction_rejects_an_active_prefix() {
        let manager = Arc::new(TargetKvRadixManager::new(2 * 64, 16).unwrap());
        let prompt = tokens(TARGET_KV_PAGE_TOKENS, 1);
        cache(&manager, &prompt);
        let active = manager.reserve(&prompt, prompt.len()).unwrap();
        assert!(manager.evict_exact_inactive_leaf(&prompt).is_err());
        drop(active);
        assert_eq!(manager.evict_exact_inactive_leaf(&prompt).unwrap(), 1);
    }

    #[test]
    fn exact_subtree_eviction_removes_every_inactive_branch() {
        let manager = Arc::new(TargetKvRadixManager::new(8 * 64, 16).unwrap());
        let common = tokens(TARGET_KV_PAGE_TOKENS, 1);
        let mut left = common.clone();
        left.extend(tokens(TARGET_KV_PAGE_TOKENS, 2));
        let mut right = common.clone();
        right.extend(tokens(TARGET_KV_PAGE_TOKENS, 3));
        cache(&manager, &left);
        cache(&manager, &right);

        assert_eq!(manager.stats().radix_nodes, 3);
        assert_eq!(
            manager
                .evict_exact_inactive_subtree_if_present(&common)
                .unwrap(),
            TargetKvExactSubtreeEviction::Evicted { pages: 3 }
        );
        let stats = manager.stats();
        assert_eq!(stats.cached_pages, 0);
        assert_eq!(stats.free_pages, 8);
        assert_eq!(stats.radix_nodes, 0);
    }

    #[test]
    fn exact_subtree_eviction_is_atomic_when_a_descendant_is_active() {
        let manager = Arc::new(TargetKvRadixManager::new(8 * 64, 16).unwrap());
        let common = tokens(TARGET_KV_PAGE_TOKENS, 1);
        let mut long = common.clone();
        long.extend(tokens(TARGET_KV_PAGE_TOKENS, 2));
        cache(&manager, &long);
        let active = manager.reserve(&long, long.len()).unwrap();

        assert!(manager
            .evict_exact_inactive_subtree_if_present(&common)
            .is_err());
        assert_eq!(manager.stats().cached_pages, 2);
        drop(active);
        assert!(matches!(
            manager
                .evict_exact_inactive_subtree_if_present(&common)
                .unwrap(),
            TargetKvExactSubtreeEviction::Evicted { pages: 2 }
        ));
    }

    #[test]
    fn split_nodes_inherit_and_release_existing_path_locks() {
        let manager = Arc::new(TargetKvRadixManager::new(6 * 64, 16).unwrap());
        let original = tokens(100, 1);
        cache(&manager, &original);
        let exact = manager.reserve(&original, original.len()).unwrap();
        let prefix = manager.reserve(&original[..70], 70).unwrap();
        assert_eq!(manager.stats().pinned_cached_pages, 2);
        drop(exact);
        assert_eq!(manager.stats().pinned_cached_pages, 2);
        drop(prefix);
        assert_eq!(manager.stats().pinned_cached_pages, 0);
    }

    #[test]
    fn leaf_lru_preserves_the_recent_branch() {
        let manager = Arc::new(TargetKvRadixManager::new(2 * 64, 16).unwrap());
        let older = tokens(TARGET_KV_PAGE_TOKENS, 1);
        let recent = tokens(TARGET_KV_PAGE_TOKENS, 2);
        cache(&manager, &older);
        cache(&manager, &recent);
        drop(manager.reserve(&recent, recent.len()).unwrap());

        let replacement = tokens(TARGET_KV_PAGE_TOKENS, 3);
        cache(&manager, &replacement);
        assert_eq!(manager.stats().evicted_pages, 1);

        let recent_hit = manager.reserve(&recent, recent.len()).unwrap();
        assert_eq!(recent_hit.matched_prefix_tokens(), recent.len());
        drop(recent_hit);
        let older_miss = manager.reserve(&older, older.len()).unwrap();
        assert_eq!(older_miss.matched_prefix_tokens(), 0);
    }

    #[test]
    fn leaf_lru_never_evicts_a_pinned_branch() {
        let manager = Arc::new(TargetKvRadixManager::new(2 * 64, 16).unwrap());
        let pinned_tokens = tokens(TARGET_KV_PAGE_TOKENS, 1);
        let inactive_tokens = tokens(TARGET_KV_PAGE_TOKENS, 2);
        cache(&manager, &pinned_tokens);
        let pinned = manager
            .reserve(&pinned_tokens, pinned_tokens.len())
            .unwrap();
        cache(&manager, &inactive_tokens);

        let replacement = tokens(TARGET_KV_PAGE_TOKENS, 3);
        cache(&manager, &replacement);
        assert_eq!(manager.stats().evicted_pages, 1);

        let pinned_hit = manager
            .reserve(&pinned_tokens, pinned_tokens.len())
            .unwrap();
        assert_eq!(
            pinned_hit.matched_prefix_tokens(),
            pinned_tokens.len(),
            "the active branch must survive pressure from an inactive replacement"
        );
        drop(pinned_hit);
        let inactive_miss = manager
            .reserve(&inactive_tokens, inactive_tokens.len())
            .unwrap();
        assert_eq!(
            inactive_miss.matched_prefix_tokens(),
            0,
            "the inactive branch must be the eviction victim"
        );
        drop(inactive_miss);
        drop(pinned);
    }

    #[test]
    fn pinned_cache_pages_and_private_credits_share_one_capacity_bound() {
        let manager = Arc::new(TargetKvRadixManager::new(4 * 64, 16).unwrap());
        let prefix = tokens(2 * 64, 1);
        cache(&manager, &prefix);
        let active = manager.reserve(&prefix, 4 * 64).unwrap();
        let stats = manager.stats();
        assert_eq!(stats.pinned_cached_pages, 2);
        assert_eq!(stats.reserved_private_pages, 2);
        assert!(manager.reserve(&tokens(1, 2), 64).is_err());
        drop(active);
        assert!(manager.reserve(&tokens(1, 2), 64).is_ok());
    }
}
