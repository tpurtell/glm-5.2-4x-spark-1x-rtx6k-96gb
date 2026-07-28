use anyhow::Result;
use glmrt_core::{NodeRole, EXPERT_HOSTS};
use glmrt_loader::resolve_snapshot;
use serde::Serialize;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use crate::cli::DoctorArgs;
use crate::{command_probe, Probe};

#[derive(Debug, Serialize)]
struct DoctorReport {
    role: String,
    model_id: String,
    hostname: Probe,
    os_kernel: Probe,
    inside_container: bool,
    cpu_arch: Probe,
    expected_cuda_arch: String,
    docker_image: DockerImageInfo,
    gpu: Probe,
    cuda_driver: Probe,
    rust: Probe,
    python: Probe,
    cmake: Probe,
    ninja: Probe,
    rdma_link: Probe,
    ibv_devices: Probe,
    hf_cache: HfCacheInfo,
    tcp_reachability: Vec<TcpReachability>,
}

#[derive(Debug, Serialize)]
struct DockerImageInfo {
    role_env: Option<String>,
    cuda_arch_env: Option<String>,
    target_platform_env: Option<String>,
    nvidia_build_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct HfCacheInfo {
    hf_home: String,
    model_cache: String,
    model_cache_visible: bool,
    snapshot_path: Option<String>,
    snapshots: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TcpReachability {
    host: String,
    dns_ok: bool,
    ssh_port_22_open: bool,
    error: Option<String>,
}

pub(crate) fn run_doctor(args: DoctorArgs) -> Result<()> {
    let role = NodeRole::from_str(&args.role)?;
    let report = collect_doctor(role, &args.model_id, args.hf_home.as_deref())?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_doctor_text(&report);
    }
    Ok(())
}

fn collect_doctor(role: NodeRole, model_id: &str, hf_home: Option<&Path>) -> Result<DoctorReport> {
    let snapshot = resolve_snapshot(model_id, hf_home)?;
    let host_list = std::env::var("GLMRT_EXPERT_HOSTS")
        .unwrap_or_else(|_| EXPERT_HOSTS.join(","))
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();

    let hf_cache = HfCacheInfo {
        hf_home: snapshot.cache_root.display().to_string(),
        model_cache: snapshot.model_cache.display().to_string(),
        model_cache_visible: snapshot.model_cache.is_dir(),
        snapshot_path: snapshot
            .snapshot_path
            .as_ref()
            .map(|path| path.display().to_string()),
        snapshots: snapshot
            .snapshots
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };

    Ok(DoctorReport {
        role: role.to_string(),
        model_id: model_id.to_owned(),
        hostname: command_probe("hostname", &[]),
        os_kernel: command_probe("uname", &["-srmo"]),
        inside_container: Path::new("/.dockerenv").exists(),
        cpu_arch: command_probe("uname", &["-m"]),
        expected_cuda_arch: role.expected_cuda_arch().to_owned(),
        docker_image: DockerImageInfo {
            role_env: std::env::var("GLMRT_ROLE").ok(),
            cuda_arch_env: std::env::var("GLMRT_CUDA_ARCH").ok(),
            target_platform_env: std::env::var("GLMRT_TARGET_PLATFORM").ok(),
            nvidia_build_id: std::env::var("NVIDIA_BUILD_ID").ok(),
        },
        gpu: command_probe(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
        ),
        cuda_driver: command_probe("nvidia-smi", &[]),
        rust: command_probe("rustc", &["--version"]),
        python: command_probe("python3", &["--version"]),
        cmake: command_probe("cmake", &["--version"]),
        ninja: command_probe("ninja", &["--version"]),
        rdma_link: command_probe("rdma", &["link"]),
        ibv_devices: command_probe("ibv_devices", &[]),
        hf_cache,
        tcp_reachability: host_list
            .into_iter()
            .map(|host| tcp_reachability(&host))
            .collect(),
    })
}

fn tcp_reachability(host: &str) -> TcpReachability {
    let mut error = None;
    let addrs = match (host, 22).to_socket_addrs() {
        Ok(addrs) => addrs.collect::<Vec<_>>(),
        Err(err) => {
            return TcpReachability {
                host: host.to_owned(),
                dns_ok: false,
                ssh_port_22_open: false,
                error: Some(err.to_string()),
            };
        }
    };
    let mut ssh_port_22_open = false;
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, Duration::from_millis(500)) {
            Ok(_) => {
                ssh_port_22_open = true;
                break;
            }
            Err(err) => error = Some(err.to_string()),
        }
    }
    TcpReachability {
        host: host.to_owned(),
        dns_ok: true,
        ssh_port_22_open,
        error: if ssh_port_22_open { None } else { error },
    }
}

fn print_doctor_text(report: &DoctorReport) {
    println!("# GLMRT doctor");
    println!("role: {}", report.role);
    println!("model_id: {}", report.model_id);
    println!("hostname: {}", report.hostname.output);
    println!("kernel: {}", report.os_kernel.output);
    println!("inside_container: {}", report.inside_container);
    println!("cpu_arch: {}", report.cpu_arch.output);
    println!("expected_cuda_arch: {}", report.expected_cuda_arch);
    println!("docker_role_env: {:?}", report.docker_image.role_env);
    println!(
        "docker_cuda_arch_env: {:?}",
        report.docker_image.cuda_arch_env
    );
    println!(
        "docker_target_platform_env: {:?}",
        report.docker_image.target_platform_env
    );
    println!("nvidia_build_id: {:?}", report.docker_image.nvidia_build_id);
    println!("hf_home: {}", report.hf_cache.hf_home);
    println!("model_cache: {}", report.hf_cache.model_cache);
    println!(
        "model_cache_visible: {}",
        report.hf_cache.model_cache_visible
    );
    println!(
        "model_snapshot: {}",
        report
            .hf_cache
            .snapshot_path
            .as_deref()
            .unwrap_or("unresolved")
    );
    print_probe("gpu", &report.gpu);
    print_probe("cuda_driver", &report.cuda_driver);
    print_probe("rust", &report.rust);
    print_probe("python", &report.python);
    print_probe("cmake", &report.cmake);
    print_probe("ninja", &report.ninja);
    print_probe("rdma_link", &report.rdma_link);
    print_probe("ibv_devices", &report.ibv_devices);
    println!("## tcp reachability");
    for host in &report.tcp_reachability {
        println!(
            "{} dns_ok={} ssh_port_22_open={} error={}",
            host.host,
            host.dns_ok,
            host.ssh_port_22_open,
            host.error.as_deref().unwrap_or("")
        );
    }
}

fn print_probe(name: &str, probe: &Probe) {
    println!("## {name}");
    println!("ok: {}", probe.ok);
    if probe.output.is_empty() {
        println!("output: ");
    } else {
        println!("{}", probe.output);
    }
}
