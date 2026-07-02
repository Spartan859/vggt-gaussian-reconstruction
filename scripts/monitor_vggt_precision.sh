#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/mnt/share/algorithm/kimi/cache/lxy/vggt-gaussian-reconstruction}"
INTERVAL="${INTERVAL:-30}"
TICKS="${TICKS:-40}"

cd "${ROOT}" || exit 1

experiments=(
    "00_baseline_quality:vggt_precision_00_baseline_quality"
    "01_anchor_frame_selection:vggt_precision_01_anchor_frame_selection"
    "02_parallax_adaptive_query:vggt_precision_02_parallax_adaptive_query"
    "03_coarse_to_fine_coarse:vggt_precision_03a_coarse"
    "04_two_stage_ba_filter:vggt_precision_04_two_stage_ba_filter"
)

for tick in $(seq 1 "${TICKS}"); do
    echo "===== $(date '+%F %T') tick=${tick}/${TICKS} ====="
    for item in "${experiments[@]}"; do
        exp="${item%%:*}"
        log_name="${item#*:}"
        scene="outputs/experiments_vggt_precision/${exp}"
        log="outputs/platform_logs/${log_name}.log"

        imgs=0
        if [[ -d "${scene}/images" ]]; then
            imgs=$(find "${scene}/images" -maxdepth 1 -type f | wc -l)
        fi
        meta=no
        [[ -f "${scene}/metadata.json" ]] && meta=yes
        sparse=no
        if [[ -e "${scene}/vggt/sparse/0/cameras.bin" || -e "${scene}/vggt/sparse/0/cameras.txt" ]]; then
            sparse=yes
        fi
        ba=no
        [[ -f "${scene}/ba/ba_stats.json" ]] && ba=yes
        ckpt=no
        if find "${scene}/runs" -path "*/gaussians_ba/checkpoint.pt" -type f -print -quit 2>/dev/null | grep -q .; then
            ckpt=yes
        fi
        eval_report=no
        if find "${scene}/runs" -name eval_report.json -type f -print -quit 2>/dev/null | grep -q .; then
            eval_report=yes
        fi

        mtime=missing
        last="no log"
        if [[ -f "${log}" ]]; then
            mtime=$(stat -c "%y" "${log}" | cut -d. -f1)
            last=$(tail -1 "${log}")
        fi

        printf "%s images=%s metadata=%s sparse=%s ba=%s ckpt=%s eval=%s log_mtime=%s\n" \
            "${exp}" "${imgs}" "${meta}" "${sparse}" "${ba}" "${ckpt}" "${eval_report}" "${mtime}"
        printf "  last: %s\n" "${last}"
    done
    sleep "${INTERVAL}"
done
