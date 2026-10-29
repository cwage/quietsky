#!/bin/sh
# Run the command as whoever owns the project mount, so files written to it
# are theirs. Under rooted Docker that is the host user's uid and gid, which
# the container sees on /app and drops to here. Under rootless Docker the
# host user is already root inside the container, /app shows as root's, and
# nothing needs doing.
set -e
owner=$(stat -c '%u:%g' /app)
if [ "$owner" = "0:0" ]; then
    exec "$@"
fi
exec setpriv --reuid "${owner%:*}" --regid "${owner#*:}" --clear-groups -- "$@"
