use std::io;

#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> io::Result<()> {
    if cpu >= libc::CPU_SETSIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "CPU index {cpu} exceeds Linux CPU affinity capacity {}",
                libc::CPU_SETSIZE
            ),
        ));
    }

    // SAFETY: cpu_set_t is valid when zero-initialized, CPU_SET receives an
    // in-range index checked above, and sched_setaffinity reads the initialized
    // set for exactly its native size. pid=0 applies to the calling thread.
    let result = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread_to_cpu(cpu: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!("pinning the current thread to CPU {cpu} requires Linux"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_cpu_outside_linux_affinity_set() {
        let error = pin_current_thread_to_cpu(libc::CPU_SETSIZE as usize)
            .expect_err("out-of-range CPU must be rejected before the system call");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
