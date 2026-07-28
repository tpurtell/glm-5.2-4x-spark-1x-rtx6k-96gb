use std::env::VarError;

#[cfg(test)]
use std::{cell::RefCell, collections::HashMap};

#[cfg(test)]
thread_local! {
    static PROBE_ENV_TEST_OVERRIDES: RefCell<HashMap<String, Option<String>>> =
        RefCell::new(HashMap::new());
}

pub(super) fn var(key: &str) -> Result<String, VarError> {
    #[cfg(test)]
    if let Some(value) = PROBE_ENV_TEST_OVERRIDES.with(|overrides| {
        let overrides = overrides.borrow();
        overrides.get(key).cloned()
    }) {
        return value.ok_or(VarError::NotPresent);
    }

    std::env::var(key)
}

pub(super) fn var_opt(key: &str) -> Option<String> {
    var(key).ok()
}

#[cfg(test)]
pub(in crate::commands::real_full) struct ProbeEnvTestOverride {
    previous: Vec<(String, Option<Option<String>>)>,
}

#[cfg(test)]
impl Drop for ProbeEnvTestOverride {
    fn drop(&mut self) {
        PROBE_ENV_TEST_OVERRIDES.with(|overrides| {
            let mut overrides = overrides.borrow_mut();
            for (key, previous) in self.previous.drain(..) {
                if let Some(previous) = previous {
                    overrides.insert(key, previous);
                } else {
                    overrides.remove(&key);
                }
            }
        });
    }
}

#[cfg(test)]
pub(in crate::commands::real_full) fn mask_for_test(keys: &[&str]) -> ProbeEnvTestOverride {
    let previous = PROBE_ENV_TEST_OVERRIDES.with(|overrides| {
        let mut overrides = overrides.borrow_mut();
        keys.iter()
            .map(|key| {
                let key = (*key).to_owned();
                let previous = overrides.insert(key.clone(), None);
                (key, previous)
            })
            .collect()
    });
    ProbeEnvTestOverride { previous }
}
