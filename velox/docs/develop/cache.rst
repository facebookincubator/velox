===========================
AsyncDataCache (File Cache)
===========================

Background
----------
Velox provides a transparent file cache (AsyncDataCache) to accelerate table scans operators through hot data reuse and prefetch algorithms. 
The file cache is integrated with the memory system to achieve dynamic memory sharing between the file cache and query memory. 
When a query fails to allocate memory, we retry the allocation by shrinking the file cache. 
Therefore, the file cache size is automatically adjusted in response to the query memory usage change. 
See `Memory Management - Velox Documentation <https://facebookincubator.github.io/velox/develop/memory.html>`_  
for more information about Velox's file cache.

Configuration Properties
------------------------
The AsyncDataCache can be enabled by setting the following config:

.. code-block:: bash

    async-data-cache-enabled=true


Other Properties
----------------
There is a ``cache.no_retention`` session property in Velox that can be set to control if a query's cached data is retained or not after its execution.

.. list-table::
   :widths: 30 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - cache.no_retention
     - bool
     - false
     - If set to true, evicts data read by a query (using a table scan) from the in-memory cache right after the access and also skips staging to the SSD cache.

Set the ``hive.node_scheduler_affinity`` session property accordingly to turn ON/OFF ``cache.no_retention``.​

.. code-block:: bash

    SET SESSION hive.node_selection_strategy='NO_PREFERENCE'; // To turn cache.no_retention ON.​
    SET SESSION hive.node_selection_strategy='SOFT_AFFINITY'; // To turn cache.no_retention OFF.​
    SET SESSION hive.node_selection_strategy='HARD_AFFINITY'; // To turn cache.no_retention OFF.​


=========
SSD Cache
=========

Background
----------
The in-memory file cache (AsyncDataCache) is configured to use SSD when provided.
The SSD serves as an extension for the async data cache (file cache).
This helps mitigate the number of reads from slower storage.

Configuration Properties
------------------------
The SSD cache can be used by setting the following configs:

.. code-block:: bash

    async-data-cache-enabled=true
    async-cache-ssd-gb=<the size of your SSD>
    async-cache-ssd-path=<path to directory that is mounted onto SSD>

.. list-table::
   :widths: 30 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - async-data-cache-enabled
     - bool
     - true
     - If true, enable async data cache.
   * - async-cache-ssd-gb
     - integer
     - 0
     - The size of the SSD.
   * - async-cache-ssd-path
     - string
     - /mnt/flash/async_cache.
     - The directory that is mounted onto SSD.


Other configuration properties can also be set to control how often the async data cache writes to SSD. 
See `Configuration Properties <../configs.rst>`_ for more SSD Cache related configuration properties.

Metrics
-------
There are SSD cache relevant metrics that Velox emits during query execution and runtime. 
See `Debugging Metrics <./debugging/metrics.rst>`_ and `Monitoring Metrics <../monitoring/metrics.rst>`_ for more details.


Setup with btrfs filesystem on worker machines (Linux only)
-----------------------------------------------------------
Multiple factors contribute to utilizing the SSD cache effectively. 
One of them is choosing the best file system that allows direct writes for best performance.
Btrfs was found to be a good file system to use due to its built-in data compression, 
support for O_DIRECT writes, and the ability to perform asynchronous discard operations. 
These features combine to enhance storage efficiency, improve performance, and optimize disk management.

NOTE: Commands below were ran successfully for worker machines of Amazon EC2 r6 instances with CentOS.


.. code-block:: bash

    # Installs the centos-release-hyperscale-experimental module and other necessary packages.
    # https://sigs.centos.org/hyperscale/content/repositories/experimental/
    # It will also upgrade the kernel to the supported version for btrfs installation.
    hostnamectl
    sudo dnf -y install centos-release-hyperscale-experimental
    sudo dnf --disablerepo=* --enablerepo=centos-hyperscale,centos-hyperscale-experimental -y update --allowerasing
    sudo dnf -y install kernel-modules-extra
    # Restart worker machine to have the new Kernel version take into effect.
    sudo shutdown -r now || true


.. code-block:: bash

    # This is for if your worker machine is part of a Docker swarm and needs to connect back to it.
    # The systemd packages need to be updated to match with the new updated kernel.
    sudo dnf -y install systemd-networkd systemd-boot


.. code-block:: bash

    # Install the btrfs packages.
    hostnamectl
    sudo yum -y install btrfs-progs
    echo "Checking /proc/filesystems for btrfs support..."
    if ! grep -q btrfs /proc/filesystems; then
        echo "Btrfs is not supported by the kernel."
        exit 1
    fi
    echo "Btrfs is supported by the kernel."


.. code-block:: bash

    # If btrfs is successfully supported by the kernel, mount btrfs onto a disk and directory path.
    sudo lsblk -d -o NAME | tail -n +2
    # Only install btrfs onto a disk that is not EBS (EBS holds the OS).
    disk_names=( $(sudo lsblk -d -o NAME | tail -n +2) )
    for disk in "${disk_names[@]}"; do
        echo "Checking disk: $disk"
        # If the disk is an Amazon EC2 NVMe Instance Storage volume, then install btrfs onto that disk
        if sudo fdisk -l "/dev/$disk" | grep -q "Amazon EC2 NVMe Instance Storage"; then
            echo "Disk $disk is an Amazon EC2 NVMe Instance Storage"
            sudo mkfs.btrfs /dev/$disk
            sudo mount -t btrfs /dev/$disk /home/centos/presto/async_data_cache
            sudo echo "/dev/$disk /home/centos/presto/async_data_cache auto noatime 0 0" | sudo tee -a /etc/fstab
            sudo lsblk -f
            break
        else
            echo "Disk $disk is not an Amazon EC2 NVMe Instance Storage volume"
        fi
    done
