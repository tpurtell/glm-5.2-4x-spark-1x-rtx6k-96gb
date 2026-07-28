use super::intermediate_sharding::{
    balanced_row_partition, ExpertIntermediateReductionDtype, ExpertIntermediateShard,
};
use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use glmrt_transport::{TcpTransportConfig, VerbsHostMappedRdmaRing, VerbsHostMappedRdmaRingConfig};
use std::{
    ffi::c_void,
    net::TcpListener,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const SPARK_RDMA_PAIR_COUNT: usize = 6;
const SPARK_RDMA_CONNECT_TIMEOUT: Duration = Duration::from_secs(300);
const SPARK_RDMA_CONNECT_RETRY: Duration = Duration::from_millis(100);
const SPARK_RDMA_FRAME_MAGIC: &[u8; 8] = b"GLMRDMA1";
const SPARK_RDMA_FRAME_VERSION: u16 = 1;
const SPARK_RDMA_TIMING_ENV: &str = "GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING";
pub(crate) const SPARK_RDMA_FRAME_HEADER_BYTES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparkExpertRdmaReductionConfig {
    pub(crate) shard: ExpertIntermediateShard,
    pub(crate) dtype: ExpertIntermediateReductionDtype,
    pub(crate) min_rows: usize,
    pub(crate) base_port: u16,
    pub(crate) rank_hosts_by_rail: Vec<Vec<String>>,
    pub(crate) rdma_devices: Vec<String>,
    pub(crate) slot_capacity_bytes: usize,
    pub(crate) ring_depth: usize,
    pub(crate) stripe_min_bytes: usize,
}

pub(crate) struct SparkExpertRdmaReduction {
    pub(crate) dtype: ExpertIntermediateReductionDtype,
    pub(crate) min_rows: usize,
    shard: ExpertIntermediateShard,
    peers: Vec<SparkRdmaPeer>,
    stripe_min_bytes: usize,
    execution_lane: u32,
    timing: bool,
}

struct SparkRdmaPeer {
    rank: usize,
    rings: Vec<VerbsHostMappedRdmaRing>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SparkRdmaFrameHeader {
    request_id: u64,
    source_rank: usize,
    destination_rank: usize,
    full_rows: usize,
    row_start: usize,
    row_count: usize,
    row_width: usize,
    row_stride_bytes: usize,
    dtype: ExpertIntermediateReductionDtype,
    payload_bytes: usize,
}

pub(crate) struct SparkExpertRdmaExchange {
    request_id: u64,
    pub(crate) row_start: usize,
    pub(crate) row_count: usize,
    received: Vec<SparkRdmaReceivedPeer>,
}

struct SparkRdmaReceivedPeer {
    rank: usize,
    rails: Vec<SparkRdmaReceivedRail>,
}

struct SparkRdmaReceivedRail {
    row_offset: usize,
    row_count: usize,
    sequence: u64,
    payload: GlmrtDeviceBuffer,
}

pub(crate) struct SparkExpertRdmaExchangeSegment {
    pub(crate) row_offset: usize,
    pub(crate) row_count: usize,
    pub(crate) peer_payloads: [GlmrtDeviceBuffer; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SparkRdmaRailTransfer {
    peer_index: usize,
    rail_index: usize,
    row_start: usize,
    row_count: usize,
    payload_bytes: usize,
}

#[derive(Clone, Copy)]
enum SparkRdmaSendRows {
    Packed(GlmrtDeviceBuffer),
    Bf16(GlmrtDeviceBuffer),
}

impl SparkExpertRdmaReduction {
    pub(crate) fn connect(
        config: SparkExpertRdmaReductionConfig,
        execution_lane: u32,
    ) -> Result<Self> {
        let ring_config =
            VerbsHostMappedRdmaRingConfig::new(config.slot_capacity_bytes, config.ring_depth)?;
        anyhow::ensure!(
            !config.rank_hosts_by_rail.is_empty(),
            "Spark RDMA reduction requires at least one rail"
        );
        let rail_count = config.rank_hosts_by_rail.len();
        for (rail_index, rank_hosts) in config.rank_hosts_by_rail.iter().enumerate() {
            anyhow::ensure!(
                rank_hosts.len() == config.shard.count,
                "Spark RDMA reduction rail {rail_index} configured {} hosts for {} ranks",
                rank_hosts.len(),
                config.shard.count
            );
        }
        anyhow::ensure!(
            config.rdma_devices.is_empty() || config.rdma_devices.len() == rail_count,
            "Spark RDMA reduction configured {} devices for {rail_count} rails",
            config.rdma_devices.len()
        );

        // Every pair owns one bidirectional RC ring per rail. The lower rank
        // connects; the higher rank listens on the deterministic pair/lane/rail port.
        let mut accepts = Vec::new();
        for peer_rank in 0..config.shard.rank {
            for rail_index in 0..rail_count {
                let port = spark_rdma_pair_rail_port(
                    config.base_port,
                    execution_lane,
                    config.shard.rank,
                    peer_rank,
                    rail_index,
                    rail_count,
                )?;
                let listener = TcpListener::bind(("0.0.0.0", port)).with_context(|| {
                    format!(
                        "binding Spark RDMA reduction rank {} lane {execution_lane} peer {peer_rank} rail {rail_index} on port {port}",
                        config.shard.rank
                    )
                })?;
                let transport = TcpTransportConfig::default();
                accepts.push((
                    peer_rank,
                    rail_index,
                    thread::Builder::new()
                        .name(format!(
                            "spark-rdma-r{}-p{}-l{}-rail{}",
                            config.shard.rank, peer_rank, execution_lane, rail_index
                        ))
                        .spawn(move || {
                            let (stream, _) = listener
                                .accept()
                                .context("accepting Spark RDMA reduction peer connection")?;
                            VerbsHostMappedRdmaRing::accept(stream, &transport)
                        })
                        .context("spawning Spark RDMA reduction accept thread")?,
                ));
            }
        }

        let transport = TcpTransportConfig::default();
        let mut rings_by_rank = (0..config.shard.count)
            .map(|_| Vec::with_capacity(rail_count))
            .collect::<Vec<_>>();
        for peer_rank in config.shard.rank + 1..config.shard.count {
            for rail_index in 0..rail_count {
                let port = spark_rdma_pair_rail_port(
                    config.base_port,
                    execution_lane,
                    config.shard.rank,
                    peer_rank,
                    rail_index,
                    rail_count,
                )?;
                let peer = format!(
                    "{}:{port}",
                    config.rank_hosts_by_rail[rail_index][peer_rank]
                );
                let rdma_device = config.rdma_devices.get(rail_index).map(String::as_str);
                let ring = connect_ring_with_retry(
                    &peer,
                    &transport,
                    ring_config,
                    rdma_device,
                )
                .with_context(|| {
                    format!(
                        "connecting Spark RDMA reduction rank {} lane {execution_lane} to rank {peer_rank} rail {rail_index} at {peer}",
                        config.shard.rank
                    )
                })?;
                rings_by_rank[peer_rank].push(ring);
            }
        }
        for (peer_rank, rail_index, accept) in accepts {
            let ring = join_ring_accept(accept).with_context(|| {
                format!(
                    "accepting Spark RDMA reduction rank {} lane {execution_lane} from rank {peer_rank} rail {rail_index}",
                    config.shard.rank
                )
            })?;
            rings_by_rank[peer_rank].push(ring);
        }
        let peers = rings_by_rank
            .into_iter()
            .enumerate()
            .filter(|(rank, _)| *rank != config.shard.rank)
            .map(|(rank, rings)| {
                anyhow::ensure!(
                    rings.len() == rail_count,
                    "Spark RDMA reduction rank {} connected {} of {rail_count} rails to rank {rank}",
                    config.shard.rank,
                    rings.len()
                );
                Ok(SparkRdmaPeer { rank, rings })
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            peers.len() + 1 == config.shard.count,
            "Spark RDMA reduction rank {} connected {} of {} peers",
            config.shard.rank,
            peers.len(),
            config.shard.count - 1
        );
        let timing = spark_rdma_timing_enabled();
        eprintln!(
            "spark_expert_rdma_reduction_ready execution_lane={} rank={} world_size={} dtype={:?} min_rows={} peers={} rails={} devices={} stripe_min_bytes={} slot_bytes={} depth={} base_port={} timing={}",
            execution_lane,
            config.shard.rank,
            config.shard.count,
            config.dtype,
            config.min_rows,
            peers.len(),
            rail_count,
            if config.rdma_devices.is_empty() {
                "auto".to_owned()
            } else {
                config.rdma_devices.join(",")
            },
            config.stripe_min_bytes,
            config.slot_capacity_bytes,
            config.ring_depth,
            config.base_port,
            timing,
        );
        Ok(Self {
            dtype: config.dtype,
            min_rows: config.min_rows,
            shard: config.shard,
            peers,
            stripe_min_bytes: config.stripe_min_bytes,
            execution_lane,
            timing,
        })
    }

    pub(crate) fn world_size(&self) -> usize {
        self.shard.count
    }

    pub(crate) fn enabled_for_rows(&self, rows: usize) -> bool {
        rows >= self.min_rows
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exchange_row_partitions(
        &mut self,
        library: &NativeLibrary,
        request_id: u64,
        packed_rows: GlmrtDeviceBuffer,
        full_rows: usize,
        row_width: usize,
        row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<SparkExpertRdmaExchange> {
        self.exchange_row_partitions_from(
            library,
            request_id,
            SparkRdmaSendRows::Packed(packed_rows),
            full_rows,
            row_width,
            row_stride_bytes,
            cuda_stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exchange_bf16_row_partitions(
        &mut self,
        library: &NativeLibrary,
        request_id: u64,
        bf16_rows: GlmrtDeviceBuffer,
        full_rows: usize,
        row_width: usize,
        row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<SparkExpertRdmaExchange> {
        anyhow::ensure!(
            self.dtype == ExpertIntermediateReductionDtype::Fp8,
            "direct BF16 Spark RDMA staging requires FP8 reduction rows"
        );
        self.exchange_row_partitions_from(
            library,
            request_id,
            SparkRdmaSendRows::Bf16(bf16_rows),
            full_rows,
            row_width,
            row_stride_bytes,
            cuda_stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn exchange_row_partitions_from(
        &mut self,
        library: &NativeLibrary,
        request_id: u64,
        send_rows: SparkRdmaSendRows,
        full_rows: usize,
        row_width: usize,
        row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<SparkExpertRdmaExchange> {
        let exchange_started = self.timing.then(Instant::now);
        anyhow::ensure!(full_rows > 0, "Spark RDMA exchange requires rows");
        anyhow::ensure!(row_width > 0, "Spark RDMA exchange requires a row width");
        anyhow::ensure!(
            row_stride_bytes > 0,
            "Spark RDMA exchange requires a row stride"
        );
        let packed_bytes = full_rows
            .checked_mul(row_stride_bytes)
            .context("Spark RDMA packed row byte count overflow")?;
        match send_rows {
            SparkRdmaSendRows::Packed(packed_rows) => anyhow::ensure!(
                !packed_rows.ptr.is_null() && packed_rows.bytes >= packed_bytes,
                "Spark RDMA packed row buffer has {} bytes, needs {packed_bytes}",
                packed_rows.bytes
            ),
            SparkRdmaSendRows::Bf16(bf16_rows) => {
                let bf16_bytes = full_rows
                    .checked_mul(row_width)
                    .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                    .context("Spark RDMA BF16 row byte count overflow")?;
                anyhow::ensure!(
                    !bf16_rows.ptr.is_null() && bf16_rows.bytes >= bf16_bytes,
                    "Spark RDMA BF16 row buffer has {} bytes, needs {bf16_bytes}",
                    bf16_rows.bytes
                );
            }
        }

        let world_size = self.shard.count;
        let rank = self.shard.rank;
        let (local_row_start, local_row_count) =
            balanced_row_partition(full_rows, world_size, rank)?;
        let local_payload_bytes = local_row_count
            .checked_mul(row_stride_bytes)
            .context("Spark RDMA local payload byte count overflow")?;

        let mut transfers = Vec::new();
        let mut send_bytes_by_rail = self
            .timing
            .then(|| vec![0_usize; self.peers.first().map_or(0, |peer| peer.rings.len())]);
        for (peer_index, peer) in self.peers.iter().enumerate() {
            let (row_start, row_count) = balanced_row_partition(full_rows, world_size, peer.rank)?;
            let payload_bytes = row_count
                .checked_mul(row_stride_bytes)
                .context("Spark RDMA peer payload byte count overflow")?;
            let active_rails = spark_rdma_active_rail_count(
                peer.rings.len(),
                row_count,
                payload_bytes,
                self.stripe_min_bytes,
            )?;
            for (rail_index, (segment_start, segment_rows)) in
                split_row_partition(row_start, row_count, active_rails)?
                    .into_iter()
                    .enumerate()
            {
                let segment_bytes = segment_rows
                    .checked_mul(row_stride_bytes)
                    .context("Spark RDMA rail payload byte count overflow")?;
                let frame_bytes = SPARK_RDMA_FRAME_HEADER_BYTES
                    .checked_add(segment_bytes)
                    .context("Spark RDMA frame byte count overflow")?;
                let ring = &peer.rings[rail_index];
                anyhow::ensure!(
                    frame_bytes <= ring.config().slot_capacity_bytes,
                    "Spark RDMA rank {rank} payload for rank {} rail {rail_index} needs {frame_bytes} bytes, slot capacity is {}",
                    peer.rank,
                    ring.config().slot_capacity_bytes
                );
                spark_rdma_send_source_slice(
                    send_rows,
                    segment_start,
                    segment_rows,
                    row_width,
                    row_stride_bytes,
                )?;
                transfers.push(SparkRdmaRailTransfer {
                    peer_index,
                    rail_index,
                    row_start: segment_start,
                    row_count: segment_rows,
                    payload_bytes: segment_bytes,
                });
                if let Some(send_bytes_by_rail) = send_bytes_by_rail.as_mut() {
                    send_bytes_by_rail[rail_index] = send_bytes_by_rail[rail_index]
                        .checked_add(segment_bytes)
                        .context("Spark RDMA timing send byte count overflow")?;
                }
            }
        }

        // Validate every frame before reserving slots. A failed operation after
        // reservation intentionally fails the request instead of reusing a
        // partially written ring slot.
        let mut send_poll_iterations = 0_u64;
        for transfer in &transfers {
            let peer = &mut self.peers[transfer.peer_index];
            let slot = if self.timing {
                let (slot, stats) = peer.rings[transfer.rail_index]
                    .reserve_send_slot_with_stats()
                    .with_context(|| {
                        format!(
                            "reserving Spark RDMA rank {rank} send slot for rank {} rail {}",
                            peer.rank, transfer.rail_index
                        )
                    })?;
                send_poll_iterations = send_poll_iterations
                    .checked_add(stats.poll_iterations)
                    .context("Spark RDMA timing send poll count overflow")?;
                slot
            } else {
                peer.rings[transfer.rail_index]
                    .reserve_send_slot()
                    .with_context(|| {
                        format!(
                            "reserving Spark RDMA rank {rank} send slot for rank {} rail {}",
                            peer.rank, transfer.rail_index
                        )
                    })?
            };
            let header = SparkRdmaFrameHeader {
                request_id,
                source_rank: rank,
                destination_rank: peer.rank,
                full_rows,
                row_start: transfer.row_start,
                row_count: transfer.row_count,
                row_width,
                row_stride_bytes,
                dtype: self.dtype,
                payload_bytes: transfer.payload_bytes,
            }
            .encode()?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    header.as_ptr(),
                    slot.host_ptr,
                    SPARK_RDMA_FRAME_HEADER_BYTES,
                );
                let target = device_buffer_slice(
                    slot.device_buffer,
                    SPARK_RDMA_FRAME_HEADER_BYTES,
                    transfer.payload_bytes,
                )?;
                match send_rows {
                    SparkRdmaSendRows::Packed(_) => {
                        let source = spark_rdma_send_source_slice(
                            send_rows,
                            transfer.row_start,
                            transfer.row_count,
                            row_width,
                            row_stride_bytes,
                        )?;
                        library
                            .copy_d2d_async(target, source, transfer.payload_bytes, cuda_stream)
                            .with_context(|| {
                                format!(
                                    "copying Spark RDMA rank {rank} rows for rank {} rail {}",
                                    peer.rank, transfer.rail_index
                                )
                            })?;
                    }
                    SparkRdmaSendRows::Bf16(_) => {
                        let source = spark_rdma_send_source_slice(
                            send_rows,
                            transfer.row_start,
                            transfer.row_count,
                            row_width,
                            row_stride_bytes,
                        )?;
                        library
                            .cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
                                source,
                                target,
                                transfer.row_count,
                                row_width,
                                row_stride_bytes,
                                cuda_stream,
                            )
                            .with_context(|| {
                                format!(
                                    "packing Spark RDMA rank {rank} BF16 rows for rank {} rail {}",
                                    peer.rank, transfer.rail_index
                                )
                            })?;
                    }
                }
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing Spark RDMA send-slot writes")?;
        }
        let staging_finished = self.timing.then(Instant::now);
        for transfer in &transfers {
            let peer = &mut self.peers[transfer.peer_index];
            peer.rings[transfer.rail_index]
                .post_reserved_send(SPARK_RDMA_FRAME_HEADER_BYTES + transfer.payload_bytes)
                .with_context(|| {
                    format!(
                        "posting Spark RDMA rank {rank} send to rank {} rail {}",
                        peer.rank, transfer.rail_index
                    )
                })?;
        }
        let posting_finished = self.timing.then(Instant::now);

        let mut received = Vec::with_capacity(self.peers.len());
        let mut recv_bytes_by_rail = send_bytes_by_rail
            .as_ref()
            .map(|bytes| vec![0_usize; bytes.len()]);
        let mut recv_poll_iterations = 0_u64;
        for peer in &mut self.peers {
            let active_rails = spark_rdma_active_rail_count(
                peer.rings.len(),
                local_row_count,
                local_payload_bytes,
                self.stripe_min_bytes,
            )?;
            let mut received_rails = Vec::with_capacity(active_rails);
            for (rail_index, (segment_start, segment_rows)) in
                split_row_partition(local_row_start, local_row_count, active_rails)?
                    .into_iter()
                    .enumerate()
            {
                let segment_bytes = segment_rows
                    .checked_mul(row_stride_bytes)
                    .context("Spark RDMA receive rail payload byte count overflow")?;
                let slot = if self.timing {
                    let (slot, stats) = peer.rings[rail_index]
                        .wait_recv_slot_with_stats()
                        .with_context(|| {
                            format!(
                                "waiting for Spark RDMA rank {rank} receive from rank {} rail {rail_index} request {request_id}",
                                peer.rank
                            )
                        })?;
                    recv_poll_iterations = recv_poll_iterations
                        .checked_add(stats.poll_iterations)
                        .context("Spark RDMA timing receive poll count overflow")?;
                    slot
                } else {
                    peer.rings[rail_index].wait_recv_slot().with_context(|| {
                        format!(
                            "waiting for Spark RDMA rank {rank} receive from rank {} rail {rail_index} request {request_id}",
                            peer.rank
                        )
                    })?
                };
                let header_bytes = unsafe {
                    std::slice::from_raw_parts(slot.host_ptr, SPARK_RDMA_FRAME_HEADER_BYTES)
                };
                let header = SparkRdmaFrameHeader::decode(header_bytes).with_context(|| {
                    format!(
                        "decoding Spark RDMA rank {rank} frame from rank {} rail {rail_index}",
                        peer.rank
                    )
                })?;
                let expected = SparkRdmaFrameHeader {
                    request_id,
                    source_rank: peer.rank,
                    destination_rank: rank,
                    full_rows,
                    row_start: segment_start,
                    row_count: segment_rows,
                    row_width,
                    row_stride_bytes,
                    dtype: self.dtype,
                    payload_bytes: segment_bytes,
                };
                anyhow::ensure!(
                    header == expected,
                    "Spark RDMA rank {rank} received unexpected frame from rank {} rail {rail_index}: expected {expected:?}, got {header:?}",
                    peer.rank
                );
                if let Some(recv_bytes_by_rail) = recv_bytes_by_rail.as_mut() {
                    recv_bytes_by_rail[rail_index] = recv_bytes_by_rail[rail_index]
                        .checked_add(segment_bytes)
                        .context("Spark RDMA timing receive byte count overflow")?;
                }
                received_rails.push(SparkRdmaReceivedRail {
                    row_offset: segment_start - local_row_start,
                    row_count: segment_rows,
                    sequence: slot.sequence,
                    payload: device_buffer_slice(
                        slot.device_buffer,
                        SPARK_RDMA_FRAME_HEADER_BYTES,
                        segment_bytes,
                    )?,
                });
            }
            received.push(SparkRdmaReceivedPeer {
                rank: peer.rank,
                rails: received_rails,
            });
        }
        if let (Some(exchange_started), Some(staging_finished), Some(posting_finished)) =
            (exchange_started, staging_finished, posting_finished)
        {
            let receive_finished = Instant::now();
            eprintln!(
                "spark_expert_rdma_exchange_timing execution_lane={} rank={} request_id={} rows={} row_stride_bytes={} send_bytes_by_rail={} recv_bytes_by_rail={} stage_us={:.3} post_us={:.3} recv_wait_us={:.3} total_us={:.3} send_poll_iterations={} recv_poll_iterations={}",
                self.execution_lane,
                rank,
                request_id,
                full_rows,
                row_stride_bytes,
                format_rail_bytes(send_bytes_by_rail.as_deref().unwrap_or_default()),
                format_rail_bytes(recv_bytes_by_rail.as_deref().unwrap_or_default()),
                elapsed_us(exchange_started, staging_finished),
                elapsed_us(staging_finished, posting_finished),
                elapsed_us(posting_finished, receive_finished),
                elapsed_us(exchange_started, receive_finished),
                send_poll_iterations,
                recv_poll_iterations,
            );
        }
        Ok(SparkExpertRdmaExchange {
            request_id,
            row_start: local_row_start,
            row_count: local_row_count,
            received,
        })
    }

    pub(crate) fn release_exchange(&mut self, exchange: SparkExpertRdmaExchange) -> Result<()> {
        anyhow::ensure!(
            exchange.received.len() == self.peers.len(),
            "Spark RDMA request {} received {} of {} peers",
            exchange.request_id,
            exchange.received.len(),
            self.peers.len()
        );
        for (peer, received) in self.peers.iter_mut().zip(exchange.received) {
            anyhow::ensure!(
                peer.rank == received.rank,
                "Spark RDMA receive rank {} does not match ring rank {}",
                received.rank,
                peer.rank
            );
            anyhow::ensure!(
                received.rails.len() <= peer.rings.len(),
                "Spark RDMA rank {} received {} rails from rank {}, but owns {} rings",
                self.shard.rank,
                received.rails.len(),
                peer.rank,
                peer.rings.len()
            );
            for (rail_index, received_rail) in received.rails.into_iter().enumerate() {
                peer.rings[rail_index]
                    .release_recv_slot(received_rail.sequence)
                    .with_context(|| {
                        format!(
                            "releasing Spark RDMA rank {} receive from rank {} rail {rail_index} request {}",
                            self.shard.rank, peer.rank, exchange.request_id
                        )
                    })?;
            }
        }
        Ok(())
    }
}

impl SparkExpertRdmaExchange {
    pub(crate) fn segments(&self) -> Result<Vec<SparkExpertRdmaExchangeSegment>> {
        anyhow::ensure!(
            self.received.len() == 3,
            "Spark RDMA exchange has {} peer payloads, expected 3",
            self.received.len()
        );
        let rail_count = self.received[0].rails.len();
        anyhow::ensure!(rail_count > 0, "Spark RDMA exchange has no received rails");
        anyhow::ensure!(
            self.received
                .iter()
                .all(|peer| peer.rails.len() == rail_count),
            "Spark RDMA exchange peer rail counts differ"
        );
        let mut segments = Vec::with_capacity(rail_count);
        let mut expected_row_offset = 0;
        for rail_index in 0..rail_count {
            let first = &self.received[0].rails[rail_index];
            anyhow::ensure!(
                first.row_offset == expected_row_offset,
                "Spark RDMA rail {rail_index} starts at row {}, expected {expected_row_offset}",
                first.row_offset
            );
            let mut peer_payloads = [GlmrtDeviceBuffer::default(); 3];
            for (payload, peer) in peer_payloads.iter_mut().zip(&self.received) {
                let received = &peer.rails[rail_index];
                anyhow::ensure!(
                    received.row_offset == first.row_offset
                        && received.row_count == first.row_count,
                    "Spark RDMA rail {rail_index} row partition differs across peers"
                );
                *payload = received.payload;
            }
            segments.push(SparkExpertRdmaExchangeSegment {
                row_offset: first.row_offset,
                row_count: first.row_count,
                peer_payloads,
            });
            expected_row_offset = expected_row_offset
                .checked_add(first.row_count)
                .context("Spark RDMA exchange row coverage overflow")?;
        }
        anyhow::ensure!(
            expected_row_offset == self.row_count,
            "Spark RDMA exchange rails cover {expected_row_offset} rows, expected {}",
            self.row_count
        );
        Ok(segments)
    }
}

impl SparkRdmaFrameHeader {
    fn encode(self) -> Result<[u8; SPARK_RDMA_FRAME_HEADER_BYTES]> {
        let mut bytes = [0_u8; SPARK_RDMA_FRAME_HEADER_BYTES];
        bytes[0..8].copy_from_slice(SPARK_RDMA_FRAME_MAGIC);
        write_u16(&mut bytes, 8, SPARK_RDMA_FRAME_VERSION);
        write_u16(
            &mut bytes,
            10,
            u16::try_from(SPARK_RDMA_FRAME_HEADER_BYTES)
                .context("Spark RDMA frame header size exceeds u16")?,
        );
        write_u32(&mut bytes, 12, u32_value(self.source_rank, "source rank")?);
        write_u32(
            &mut bytes,
            16,
            u32_value(self.destination_rank, "destination rank")?,
        );
        write_u32(&mut bytes, 20, u32_value(self.full_rows, "full rows")?);
        write_u32(&mut bytes, 24, u32_value(self.row_start, "row start")?);
        write_u32(&mut bytes, 28, u32_value(self.row_count, "row count")?);
        write_u32(&mut bytes, 32, u32_value(self.row_width, "row width")?);
        write_u32(
            &mut bytes,
            36,
            u32_value(self.row_stride_bytes, "row stride")?,
        );
        write_u32(&mut bytes, 40, reduction_dtype_code(self.dtype));
        write_u64(&mut bytes, 48, self.request_id);
        write_u64(
            &mut bytes,
            56,
            u64::try_from(self.payload_bytes).context("Spark RDMA payload exceeds u64")?,
        );
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() >= SPARK_RDMA_FRAME_HEADER_BYTES,
            "Spark RDMA frame header has {} bytes, needs {}",
            bytes.len(),
            SPARK_RDMA_FRAME_HEADER_BYTES
        );
        anyhow::ensure!(
            &bytes[0..8] == SPARK_RDMA_FRAME_MAGIC,
            "Spark RDMA frame magic mismatch"
        );
        anyhow::ensure!(
            read_u16(bytes, 8) == SPARK_RDMA_FRAME_VERSION,
            "Spark RDMA frame version {} is unsupported",
            read_u16(bytes, 8)
        );
        anyhow::ensure!(
            read_u16(bytes, 10) as usize == SPARK_RDMA_FRAME_HEADER_BYTES,
            "Spark RDMA frame header size {} is unsupported",
            read_u16(bytes, 10)
        );
        Ok(Self {
            source_rank: read_u32(bytes, 12) as usize,
            destination_rank: read_u32(bytes, 16) as usize,
            full_rows: read_u32(bytes, 20) as usize,
            row_start: read_u32(bytes, 24) as usize,
            row_count: read_u32(bytes, 28) as usize,
            row_width: read_u32(bytes, 32) as usize,
            row_stride_bytes: read_u32(bytes, 36) as usize,
            dtype: reduction_dtype_from_code(read_u32(bytes, 40))?,
            request_id: read_u64(bytes, 48),
            payload_bytes: usize::try_from(read_u64(bytes, 56))
                .context("Spark RDMA payload size exceeds usize")?,
        })
    }
}

fn reduction_dtype_code(dtype: ExpertIntermediateReductionDtype) -> u32 {
    match dtype {
        ExpertIntermediateReductionDtype::Bf16 => 1,
        ExpertIntermediateReductionDtype::Fp8 => 2,
        ExpertIntermediateReductionDtype::Nvfp4 => 3,
    }
}

fn reduction_dtype_from_code(code: u32) -> Result<ExpertIntermediateReductionDtype> {
    match code {
        1 => Ok(ExpertIntermediateReductionDtype::Bf16),
        2 => Ok(ExpertIntermediateReductionDtype::Fp8),
        3 => Ok(ExpertIntermediateReductionDtype::Nvfp4),
        _ => anyhow::bail!("unsupported Spark RDMA dtype code {code}"),
    }
}

fn u32_value(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("Spark RDMA {label} exceeds u32"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn device_buffer_slice(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .context("Spark RDMA device buffer slice overflow")?;
    anyhow::ensure!(
        !buffer.ptr.is_null() && end <= buffer.bytes,
        "Spark RDMA device buffer slice [{offset_bytes}, {end}) exceeds {} bytes",
        buffer.bytes
    );
    Ok(GlmrtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn spark_rdma_send_source_slice(
    send_rows: SparkRdmaSendRows,
    row_start: usize,
    row_count: usize,
    row_width: usize,
    row_stride_bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    let (buffer, source_row_bytes, context) = match send_rows {
        SparkRdmaSendRows::Packed(buffer) => (buffer, row_stride_bytes, "Spark RDMA packed source"),
        SparkRdmaSendRows::Bf16(buffer) => (
            buffer,
            row_width
                .checked_mul(std::mem::size_of::<u16>())
                .context("Spark RDMA BF16 source row byte count overflow")?,
            "Spark RDMA BF16 source",
        ),
    };
    let offset = row_start
        .checked_mul(source_row_bytes)
        .with_context(|| format!("{context} row offset overflow"))?;
    let bytes = row_count
        .checked_mul(source_row_bytes)
        .with_context(|| format!("{context} byte count overflow"))?;
    device_buffer_slice(buffer, offset, bytes)
}

fn connect_ring_with_retry(
    peer: &str,
    transport: &TcpTransportConfig,
    config: VerbsHostMappedRdmaRingConfig,
    rdma_device: Option<&str>,
) -> Result<VerbsHostMappedRdmaRing> {
    let started = Instant::now();
    let mut last_error = None;
    while started.elapsed() < SPARK_RDMA_CONNECT_TIMEOUT {
        let result = match rdma_device {
            Some(device) => {
                VerbsHostMappedRdmaRing::connect_on_device(peer, transport, config, device)
            }
            None => VerbsHostMappedRdmaRing::connect(peer, transport, config),
        };
        match result {
            Ok(ring) => return Ok(ring),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(SPARK_RDMA_CONNECT_RETRY);
            }
        }
    }
    anyhow::bail!(
        "timed out connecting mapped Spark RDMA ring to {peer}: {}",
        last_error
            .map(|error| format!("{error:#}"))
            .unwrap_or_else(|| "no connection attempt completed".to_owned())
    )
}

fn spark_rdma_active_rail_count(
    available_rails: usize,
    rows: usize,
    payload_bytes: usize,
    stripe_min_bytes: usize,
) -> Result<usize> {
    anyhow::ensure!(available_rails > 0, "Spark RDMA peer has no rails");
    anyhow::ensure!(rows > 0, "Spark RDMA row partition is empty");
    anyhow::ensure!(stripe_min_bytes > 0, "Spark RDMA stripe threshold is zero");
    Ok(
        if available_rails > 1 && payload_bytes >= stripe_min_bytes {
            available_rails.min(rows)
        } else {
            1
        },
    )
}

fn spark_rdma_timing_enabled() -> bool {
    std::env::var(SPARK_RDMA_TIMING_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_us(start: Instant, end: Instant) -> f64 {
    end.duration_since(start).as_secs_f64() * 1_000_000.0
}

fn format_rail_bytes(bytes: &[usize]) -> String {
    bytes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn split_row_partition(
    row_start: usize,
    row_count: usize,
    rail_count: usize,
) -> Result<Vec<(usize, usize)>> {
    anyhow::ensure!(row_count > 0, "Spark RDMA row partition is empty");
    anyhow::ensure!(
        rail_count > 0 && rail_count <= row_count,
        "Spark RDMA rail count {rail_count} is invalid for {row_count} rows"
    );
    (0..rail_count)
        .map(|rail_index| {
            let (relative_start, rows) = balanced_row_partition(row_count, rail_count, rail_index)?;
            let start = row_start
                .checked_add(relative_start)
                .context("Spark RDMA rail row start overflow")?;
            Ok((start, rows))
        })
        .collect()
}

fn join_ring_accept(
    accept: JoinHandle<Result<VerbsHostMappedRdmaRing>>,
) -> Result<VerbsHostMappedRdmaRing> {
    accept
        .join()
        .map_err(|_| anyhow::anyhow!("Spark RDMA reduction accept thread panicked"))?
}

fn spark_rdma_pair_index(rank_a: usize, rank_b: usize) -> Result<usize> {
    anyhow::ensure!(
        rank_a < 4 && rank_b < 4 && rank_a != rank_b,
        "invalid Spark RDMA rank pair {rank_a}/{rank_b}"
    );
    let (lower, higher) = if rank_a < rank_b {
        (rank_a, rank_b)
    } else {
        (rank_b, rank_a)
    };
    let mut index = 0;
    for left in 0..4 {
        for right in left + 1..4 {
            if (left, right) == (lower, higher) {
                return Ok(index);
            }
            index += 1;
        }
    }
    unreachable!("validated four-rank pair must be present")
}

fn spark_rdma_pair_rail_port(
    base_port: u16,
    execution_lane: u32,
    rank_a: usize,
    rank_b: usize,
    rail_index: usize,
    rail_count: usize,
) -> Result<u16> {
    anyhow::ensure!(
        rail_count > 0 && rail_index < rail_count,
        "invalid Spark RDMA rail {rail_index}/{rail_count}"
    );
    let lane = usize::try_from(execution_lane).context("Spark RDMA lane exceeds usize")?;
    let offset = lane
        .checked_mul(SPARK_RDMA_PAIR_COUNT)
        .and_then(|offset| offset.checked_mul(rail_count))
        .and_then(|offset| {
            spark_rdma_pair_index(rank_a, rank_b)
                .ok()
                .and_then(|pair| pair.checked_mul(rail_count))
                .and_then(|pair_offset| offset.checked_add(pair_offset))
        })
        .and_then(|offset| offset.checked_add(rail_index))
        .context("Spark RDMA pair/rail port offset overflow")?;
    base_port
        .checked_add(u16::try_from(offset).context("Spark RDMA pair/rail port offset exceeds u16")?)
        .context("Spark RDMA pair/rail port overflow")
}

#[cfg(test)]
mod tests {
    use super::{
        spark_rdma_active_rail_count, spark_rdma_pair_index, spark_rdma_pair_rail_port,
        split_row_partition, SparkExpertRdmaExchange, SparkRdmaFrameHeader, SparkRdmaReceivedPeer,
        SparkRdmaReceivedRail, SPARK_RDMA_FRAME_HEADER_BYTES,
    };
    use crate::commands::real_full::intermediate_sharding::ExpertIntermediateReductionDtype;
    use glmrt_ffi::GlmrtDeviceBuffer;

    #[test]
    fn pair_ports_are_symmetric_and_unique_per_lane() {
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (expected, (left, right)) in pairs.into_iter().enumerate() {
            assert_eq!(spark_rdma_pair_index(left, right).unwrap(), expected);
            assert_eq!(spark_rdma_pair_index(right, left).unwrap(), expected);
            assert_eq!(
                spark_rdma_pair_rail_port(9_300, 0, left, right, 0, 1).unwrap(),
                9_300 + expected as u16
            );
            assert_eq!(
                spark_rdma_pair_rail_port(9_300, 1, left, right, 0, 1).unwrap(),
                9_306 + expected as u16
            );
            assert_eq!(
                spark_rdma_pair_rail_port(9_300, 0, left, right, 0, 2).unwrap(),
                9_300 + (expected * 2) as u16
            );
            assert_eq!(
                spark_rdma_pair_rail_port(9_300, 0, left, right, 1, 2).unwrap(),
                9_301 + (expected * 2) as u16
            );
            assert_eq!(
                spark_rdma_pair_rail_port(9_300, 1, left, right, 0, 2).unwrap(),
                9_312 + (expected * 2) as u16
            );
        }
        assert!(spark_rdma_pair_index(0, 0).is_err());
        assert!(spark_rdma_pair_rail_port(u16::MAX, 1, 0, 1, 0, 2).is_err());
        assert!(spark_rdma_pair_rail_port(9_300, 0, 0, 1, 2, 2).is_err());
    }

    #[test]
    fn large_row_partitions_stripe_without_padding() {
        assert_eq!(
            spark_rdma_active_rail_count(2, 5, 512 * 1024, 256 * 1024).unwrap(),
            2
        );
        assert_eq!(
            split_row_partition(100, 5, 2).unwrap(),
            vec![(100, 3), (103, 2)]
        );
        assert_eq!(
            spark_rdma_active_rail_count(2, 5, 64 * 1024, 256 * 1024).unwrap(),
            1
        );
        assert_eq!(split_row_partition(100, 5, 1).unwrap(), vec![(100, 5)]);
    }

    #[test]
    fn exchange_segments_align_peer_payloads_by_rail() {
        let received = (0..3)
            .map(|rank| SparkRdmaReceivedPeer {
                rank,
                rails: vec![
                    SparkRdmaReceivedRail {
                        row_offset: 0,
                        row_count: 3,
                        sequence: 10,
                        payload: GlmrtDeviceBuffer::default(),
                    },
                    SparkRdmaReceivedRail {
                        row_offset: 3,
                        row_count: 2,
                        sequence: 11,
                        payload: GlmrtDeviceBuffer::default(),
                    },
                ],
            })
            .collect();
        let exchange = SparkExpertRdmaExchange {
            request_id: 7,
            row_start: 100,
            row_count: 5,
            received,
        };

        let segments = exchange.segments().unwrap();

        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].row_offset, segments[0].row_count), (0, 3));
        assert_eq!((segments[1].row_offset, segments[1].row_count), (3, 2));
    }

    #[test]
    fn frame_header_round_trips_without_native_layout_assumptions() {
        let header = SparkRdmaFrameHeader {
            request_id: 0x1020_3040_5060_7080,
            source_rank: 3,
            destination_rank: 1,
            full_rows: 1_007,
            row_start: 252,
            row_count: 252,
            row_width: 6_144,
            row_stride_bytes: 6_148,
            dtype: ExpertIntermediateReductionDtype::Fp8,
            payload_bytes: 252 * 6_148,
        };
        let encoded = header.encode().unwrap();
        assert_eq!(encoded.len(), SPARK_RDMA_FRAME_HEADER_BYTES);
        assert_eq!(SparkRdmaFrameHeader::decode(&encoded).unwrap(), header);
    }
}
