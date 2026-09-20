# ==============================================================================
# AURA-DRIVE™ 2026: HARDENED LINUX APPARMOR PROFILE
# ==============================================================================
# Restricts container processes from unauthorized syscalls and file operations
# ==============================================================================
#include <tunables/global>

profile aura-drive-hardened flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/nameservice>

  # Allow read access to Python standard libraries and system binaries
  /usr/bin/python3* ix,
  /usr/local/lib/python3*/** r,
  /usr/lib/python3*/** r,
  /lib/** r,
  /usr/lib/** r,

  # Allow read-only access to AURA application files
  /app/** r,
  /app/models/** r,
  /etc/sros2/keystore/** r,

  # Restrict write operations exclusively to designated tmpfs
  /tmp/** rw,
  /dev/shm/** rw,
  /dev/null rw,
  /dev/zero r,
  /dev/urandom r,

  # Deny raw network sockets and unauthorized capabilities
  deny network raw,
  deny capability sys_admin,
  deny capability sys_ptrace,
  deny capability sys_rawio,
  deny capability dac_override,
}

