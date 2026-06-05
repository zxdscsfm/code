#!/usr/bin/env bash

MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-80}"
MONITOR_EARLY_CLIENT_GRACE_SECONDS="${MONITOR_EARLY_CLIENT_GRACE_SECONDS:-90}"
MONITOR_SERVER_PID=""
declare -ag MONITOR_ALL_PIDS=()
declare -Ag MONITOR_ROLE=()
declare -Ag MONITOR_LABEL=()
declare -Ag MONITOR_LOG=()

print_log_tail() {
    local log_file="$1"
    if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
        echo "----- tail -${MONITOR_TAIL_LINES} ${log_file} -----"
        tail -n "${MONITOR_TAIL_LINES}" "${log_file}" || true
        echo "----- end tail ${log_file} -----"
    else
        echo "Log file not found: ${log_file}"
    fi
}

cleanup_monitored_processes() {
    local pid
    trap - INT TERM
    for pid in "${MONITOR_ALL_PIDS[@]}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
            kill "${pid}" >/dev/null 2>&1 || true
        fi
    done
    sleep 5
    for pid in "${MONITOR_ALL_PIDS[@]}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
            kill -9 "${pid}" >/dev/null 2>&1 || true
        fi
    done
}

on_monitor_termination() {
    echo "[monitor] received termination signal; cleaning up child processes"
    cleanup_monitored_processes
    exit 143
}

register_monitored_process() {
    local pid="$1"
    local role="$2"
    local label="$3"
    local log_file="$4"
    MONITOR_ALL_PIDS+=("${pid}")
    MONITOR_ROLE["${pid}"]="${role}"
    MONITOR_LABEL["${pid}"]="${label}"
    MONITOR_LOG["${pid}"]="${log_file}"
    echo "[monitor] started ${role}: pid=${pid} ${label} log=${log_file}"
}

init_run_monitor() {
    local server_pid="$1"
    local server_label="$2"
    local server_log="$3"
    MONITOR_SERVER_PID="${server_pid}"
    register_monitored_process "${server_pid}" "server" "${server_label}" "${server_log}"
    trap on_monitor_termination INT TERM
}

launch_flower_client() {
    local cid="$1"
    local client_name="$2"
    local sup_type="$3"
    local gpu="$4"
    local log_file="$5"
    python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} \
        --role client --cid "${cid}" --client "${client_name}" --sup_type "${sup_type}" --gpu "${gpu}" \
        > "${log_file}" 2>&1 &
    local pid=$!
    register_monitored_process "${pid}" "client" "cid=${cid} client=${client_name} sup_type=${sup_type} gpu=${gpu}" "${log_file}"
}

monitor_flower_processes() {
    local server_done=0
    local done_pid=""
    local status=0
    local role=""
    local label=""
    local log_file=""

    while true; do
        done_pid=""
        set +e
        wait -n -p done_pid
        status=$?
        set -e

        if [ "${status}" -eq 127 ]; then
            break
        fi

        role="${MONITOR_ROLE[${done_pid}]:-unknown}"
        label="${MONITOR_LABEL[${done_pid}]:-pid=${done_pid}}"
        log_file="${MONITOR_LOG[${done_pid}]:-}"
        echo "[monitor] exited: role=${role} status=${status} ${label}"

        if [ "${role}" = "server" ]; then
            server_done=1
            if [ "${status}" -ne 0 ]; then
                echo "[monitor] ERROR: server exited with nonzero status ${status}"
                print_log_tail "${log_file}"
                cleanup_monitored_processes
                exit "${status}"
            fi
            echo "[monitor] server exited normally; waiting for clients to close"
            continue
        fi

        if [ "${role}" = "client" ] && [ "${server_done}" -eq 0 ]; then
            echo "[monitor] client exited before server; waiting ${MONITOR_EARLY_CLIENT_GRACE_SECONDS}s to distinguish normal shutdown from a stuck server"
            local timer_pid=""
            local grace_pid=""
            local grace_status=0
            (sleep "${MONITOR_EARLY_CLIENT_GRACE_SECONDS}") &
            timer_pid=$!

            grace_pid=""
            set +e
            wait -n -p grace_pid "${MONITOR_SERVER_PID}" "${timer_pid}"
            grace_status=$?
            set -e

            if [ "${grace_pid}" = "${MONITOR_SERVER_PID}" ]; then
                server_done=1
                kill "${timer_pid}" >/dev/null 2>&1 || true
                wait "${timer_pid}" >/dev/null 2>&1 || true
                if [ "${grace_status}" -ne 0 ]; then
                    echo "[monitor] ERROR: server exited with nonzero status ${grace_status} after early client exit"
                    print_log_tail "${MONITOR_LOG[${MONITOR_SERVER_PID}]:-}"
                    cleanup_monitored_processes
                    exit "${grace_status}"
                fi
                echo "[monitor] server completed during grace window; treating early client exit as normal shutdown"
                continue
            fi

            echo "[monitor] ERROR: client exited before server completed and server did not finish within grace window"
            print_log_tail "${log_file}"
            cleanup_monitored_processes
            if [ "${status}" -eq 0 ]; then
                exit 1
            fi
            exit "${status}"
        fi

        if [ "${role}" = "client" ] && [ "${status}" -ne 0 ]; then
            echo "[monitor] ERROR: client exited with nonzero status ${status}"
            print_log_tail "${log_file}"
            cleanup_monitored_processes
            exit "${status}"
        fi
    done
}
