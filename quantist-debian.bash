# How to Prepare WSL Environment for Quantist API
# Reference: https://github.com/docker-library/python/blob/de17a909b9143e7715550ce85023fee87c48c7d6/3.11/slim-bookworm/Dockerfile

# First, install distro Debian Bookworm

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PATH=/usr/local/bin:$PATH
export LANG=C.UTF-8

sudo apt-get update && apt-get upgrade -y
apt-get install -y --no-install-recommends ca-certificates netbase tzdata
apt-get install -y --no-install-recommends dpkg-dev gcc gnupg libbluetooth-dev libbz2-dev libc6-dev libdb-dev libexpat1-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev make tk-dev uuid-dev wget xz-utils zlib1g-dev

sudo apt-get install -y git
sudo apt-get install -y postgresql postgresql-contrib
sudo apt-get python3.11.9 python3-pip python3-venv

rm -rf /var/lib/apt/lists/*

# =====
sudo nano /etc/resolv.conf
# input (without comment)
# [boot]
# systemd=true
