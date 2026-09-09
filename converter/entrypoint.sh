#!/bin/sh
# MEDUSA 2026 .d converter sidecar.
#
# Watches JOBS_DIR for <file_id>.d.req job files written by the web service
# and runs msconvert (Proteowizard under wine) for each one:
#
#   <file_id>.d.req            input=<abs dir of extracted .d>
#                              outfile=<basename for the mzXML output>
#                              outdir=<abs dir for the output>
#
# Result: msconvert writes <outdir>/<outfile>.mzXML.gz (-g) and we record
# <file_id>.done ("exit=<code>" + stderr tail) which the web service polls.
# Jobs are processed strictly sequentially (wine is heavy, one at a time).

JOBS_DIR="${JOBS_DIR:-/data/uploads/jobs}"
export WINEDEBUG=-all

mkdir -p "$JOBS_DIR"
echo "[converter] watching $JOBS_DIR"

while true; do
    for req in "$JOBS_DIR"/*.d.req; do
        [ -e "$req" ] || continue
        id="${req##*/}"
        id="${id%.d.req}"

        input=$(grep '^input=' "$req" | cut -d= -f2-)
        outfile=$(grep '^outfile=' "$req" | cut -d= -f2-)
        outdir=$(grep '^outdir=' "$req" | cut -d= -f2-)

        echo "[converter] $id: msconvert $input -> $outdir/$outfile"
        log="$JOBS_DIR/$id.log"
        wine msconvert "$input" --mzXML --64 -g --zlib \
            --outfile "$outfile" -o "$outdir" >"$log" 2>&1
        code=$?

        # .done before .req removal: a crash in between just re-runs the job
        {
            echo "exit=$code"
            echo "log_tail:"
            tail -n 20 "$log"
        } > "$JOBS_DIR/$id.done"
        rm -f "$req"
        echo "[converter] $id: finished (exit=$code)"
    done
    touch "$JOBS_DIR/.heartbeat"
    sleep 2
done
