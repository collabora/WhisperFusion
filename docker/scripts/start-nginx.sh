#!/bin/bash
echo "Starting web service with Nginx"
# Then execute the original Nginx entrypoint with the command passed to the script
exec "/usr/sbin/nginx" "-g" "daemon off;" >/dev/null 2>&1
