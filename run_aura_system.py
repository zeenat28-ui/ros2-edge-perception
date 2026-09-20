#!/usr/bin/env python3
"""
Backward compatibility entrypoint for experimental system demonstrator.

NOTE: This script forwards to `scripts.experimental_system_demonstrator`.
For production ROS 2 perception and tracking nodes, refer directly to
`ros2_edge_perception.perception_node`.
"""

from scripts.experimental_system_demonstrator import MasterAURASystem, main

if __name__ == "__main__":
    main()
