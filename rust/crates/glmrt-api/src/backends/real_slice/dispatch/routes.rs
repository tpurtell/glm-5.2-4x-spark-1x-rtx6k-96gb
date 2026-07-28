use glmrt_core::RouteEntry;

use crate::{runtime_error, ApiError, RealSlicePrefillRouteRow, RealSliceRoute};

pub(super) struct RealRouterRouteGroup {
    pub(super) owner: String,
    pub(super) routes: Vec<RouteEntry>,
}

pub(super) struct RealPrefillRouteGroup {
    pub(super) owner: String,
    pub(super) rows: Vec<RealPrefillGroupedRow>,
}

pub(super) struct RealPrefillGroupedRow {
    pub(super) row_index: usize,
    pub(super) row_id: u64,
    pub(super) routes: Vec<RouteEntry>,
}

pub(super) fn partition_real_prefill_routes(
    route_rows: &[RealSlicePrefillRouteRow],
) -> Vec<RealPrefillRouteGroup> {
    let mut groups: Vec<RealPrefillRouteGroup> = Vec::new();
    for (row_index, row) in route_rows.iter().enumerate() {
        for route in &row.routes {
            let entry = RouteEntry {
                expert_id: route.expert_id,
                gate: route.normalized_weight,
            };
            let group = match groups.iter_mut().find(|group| group.owner == route.owner) {
                Some(group) => group,
                None => {
                    groups.push(RealPrefillRouteGroup {
                        owner: route.owner.clone(),
                        rows: Vec::new(),
                    });
                    groups.last_mut().expect("just pushed prefill route group")
                }
            };
            if let Some(grouped_row) = group
                .rows
                .iter_mut()
                .find(|grouped_row| grouped_row.row_id == row.row_id)
            {
                grouped_row.routes.push(entry);
            } else {
                group.rows.push(RealPrefillGroupedRow {
                    row_index,
                    row_id: row.row_id,
                    routes: vec![entry],
                });
            }
        }
    }
    groups
}

pub(super) fn partition_real_router_routes(routes: &[RealSliceRoute]) -> Vec<RealRouterRouteGroup> {
    let mut groups: Vec<RealRouterRouteGroup> = Vec::new();
    for route in routes {
        let entry = RouteEntry {
            expert_id: route.expert_id,
            gate: route.normalized_weight,
        };
        if let Some(group) = groups.iter_mut().find(|group| group.owner == route.owner) {
            group.routes.push(entry);
        } else {
            groups.push(RealRouterRouteGroup {
                owner: route.owner.clone(),
                routes: vec![entry],
            });
        }
    }
    groups
}

pub(super) fn target_for_real_router_owner(
    targets: &[String],
    owner: &str,
) -> Result<String, ApiError> {
    if targets.len() == 1 {
        if let Some((target_owner, target_addr)) = targets[0].split_once('=') {
            if target_owner == owner {
                return Ok(target_addr.to_owned());
            }
            return Err(missing_real_router_owner_target(owner));
        }
        return Ok(targets[0].clone());
    }
    for target in targets {
        if target == owner {
            return Ok(target.clone());
        }
        if let Some((target_owner, target_addr)) = target.split_once('=') {
            if target_owner == owner {
                return Ok(target_addr.to_owned());
            }
        }
    }
    Err(missing_real_router_owner_target(owner))
}

fn missing_real_router_owner_target(owner: &str) -> ApiError {
    runtime_error(format!(
        "real-glm-slice TCP dispatch has no expert target for owner {owner}; pass one target or owner=host:port entries"
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        partition_real_prefill_routes, partition_real_router_routes, target_for_real_router_owner,
    };
    use crate::{RealSlicePrefillRouteRow, RealSliceRoute};

    fn route(owner: &str, expert_id: u32, normalized_weight: f32) -> RealSliceRoute {
        RealSliceRoute {
            expert_id,
            owner: owner.to_owned(),
            score: normalized_weight,
            corrected_score: normalized_weight,
            normalized_weight,
        }
    }

    #[test]
    fn real_slice_router_routes_group_by_owner_in_order() {
        let groups = partition_real_router_routes(&[
            route("ostrich", 3, 0.25),
            route("dodo", 9, 0.5),
            route("ostrich", 11, 0.75),
        ]);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].owner, "ostrich");
        assert_eq!(groups[0].routes.len(), 2);
        assert_eq!(groups[0].routes[0].expert_id, 3);
        assert_eq!(groups[0].routes[1].expert_id, 11);
        assert_eq!(groups[1].owner, "dodo");
        assert_eq!(groups[1].routes[0].gate, 0.5);
    }

    #[test]
    fn real_slice_prefill_routes_preserve_row_identity_per_owner() {
        let groups = partition_real_prefill_routes(&[
            RealSlicePrefillRouteRow {
                row_id: 100,
                token_id: 7,
                routes: vec![route("kiwi", 1, 0.2), route("emu", 2, 0.3)],
            },
            RealSlicePrefillRouteRow {
                row_id: 101,
                token_id: 8,
                routes: vec![route("kiwi", 4, 0.5)],
            },
            RealSlicePrefillRouteRow {
                row_id: 100,
                token_id: 7,
                routes: vec![route("kiwi", 6, 0.7)],
            },
        ]);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].owner, "kiwi");
        assert_eq!(groups[0].rows.len(), 2);
        assert_eq!(groups[0].rows[0].row_index, 0);
        assert_eq!(groups[0].rows[0].row_id, 100);
        assert_eq!(groups[0].rows[0].routes.len(), 2);
        assert_eq!(groups[0].rows[0].routes[1].expert_id, 6);
        assert_eq!(groups[0].rows[1].row_index, 1);
        assert_eq!(groups[1].owner, "emu");
        assert_eq!(groups[1].rows[0].row_id, 100);
    }

    #[test]
    fn real_slice_owner_targets_accept_single_or_owner_mapped_entries() {
        assert_eq!(
            target_for_real_router_owner(&["127.0.0.1:9141".to_owned()], "kiwi").unwrap(),
            "127.0.0.1:9141"
        );
        assert_eq!(
            target_for_real_router_owner(&["kiwi=10.0.0.11:9141".to_owned()], "kiwi").unwrap(),
            "10.0.0.11:9141"
        );
        assert_eq!(
            target_for_real_router_owner(
                &[
                    "kiwi=10.0.0.11:9141".to_owned(),
                    "emu=10.0.0.12:9141".to_owned()
                ],
                "emu",
            )
            .unwrap(),
            "10.0.0.12:9141"
        );
        assert!(target_for_real_router_owner(&["kiwi=10.0.0.11:9141".to_owned()], "emu",).is_err());
        assert!(target_for_real_router_owner(
            &[
                "kiwi=10.0.0.11:9141".to_owned(),
                "dodo=10.0.0.13:9141".to_owned()
            ],
            "emu",
        )
        .is_err());
    }
}
