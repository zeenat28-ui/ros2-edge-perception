"""
AURA-Drive™ ROS 2 Managed Lifecycle State Machine
=================================================
Deterministic lifecycle state management adhering to ROS 2 Design & rclpy.lifecycle:
  - States: UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED.
  - Transitions: configure(), activate(), deactivate(), cleanup(), shutdown().
  - Deterministic fault detection with automatic rollback to safe INACTIVE state.
"""

import enum
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Lifecycle] %(message)s")
logger = logging.getLogger("AURA_Lifecycle")


class LifecycleState(enum.Enum):
    PRIMARY_UNCONFIGURED = "UNCONFIGURED"
    PRIMARY_INACTIVE = "INACTIVE"
    PRIMARY_ACTIVE = "ACTIVE"
    PRIMARY_FINALIZED = "FINALIZED"

    # Transition states
    TRANSITION_CONFIGURING = "CONFIGURING"
    TRANSITION_CLEANINGUP = "CLEANINGUP"
    TRANSITION_ACTIVATING = "ACTIVATING"
    TRANSITION_DEACTIVATING = "DEACTIVATING"
    TRANSITION_SHUTTINGDOWN = "SHUTTINGDOWN"
    TRANSITION_ERRORPROCESSING = "ERRORPROCESSING"


class TransitionReturn(enum.Enum):
    SUCCESS = 0
    FAILURE = 1
    ERROR = 2


@dataclass
class ManagedNode:
    node_name: str
    current_state: LifecycleState = LifecycleState.PRIMARY_UNCONFIGURED
    on_configure_cb: Optional[Callable[[], bool]] = None
    on_activate_cb: Optional[Callable[[], bool]] = None
    on_deactivate_cb: Optional[Callable[[], bool]] = None
    on_cleanup_cb: Optional[Callable[[], bool]] = None
    on_shutdown_cb: Optional[Callable[[], bool]] = None
    transition_history: List[str] = field(default_factory=list)


class ROS2LifecycleManager:
    """
    Orchestrates deterministic lifecycle transitions across the AURA-Drive autonomy stack.
    """

    def __init__(self):
        self._managed_nodes: Dict[str, ManagedNode] = {}

    def register_node(
        self,
        node_name: str,
        on_configure: Optional[Callable[[], bool]] = None,
        on_activate: Optional[Callable[[], bool]] = None,
        on_deactivate: Optional[Callable[[], bool]] = None,
        on_cleanup: Optional[Callable[[], bool]] = None,
        on_shutdown: Optional[Callable[[], bool]] = None,
    ) -> ManagedNode:
        """Register a node under lifecycle supervision."""
        node = ManagedNode(
            node_name=node_name,
            current_state=LifecycleState.PRIMARY_UNCONFIGURED,
            on_configure_cb=on_configure,
            on_activate_cb=on_activate,
            on_deactivate_cb=on_deactivate,
            on_cleanup_cb=on_cleanup,
            on_shutdown_cb=on_shutdown,
        )
        self._managed_nodes[node_name] = node
        logger.info(f"Registered lifecycle node: {node_name} (State: {node.current_state.value})")
        return node

    def trigger_transition(self, node_name: str, target_transition: str) -> TransitionReturn:
        """
        Executes a formal lifecycle transition with deterministic rollback on failure.
        """
        if node_name not in self._managed_nodes:
            logger.error(f"Cannot transition unknown node: {node_name}")
            return TransitionReturn.ERROR

        node = self._managed_nodes[node_name]
        start_state = node.current_state

        if target_transition == "configure":
            if start_state != LifecycleState.PRIMARY_UNCONFIGURED:
                logger.warning(f"{node_name}: Cannot configure from {start_state.value}")
                return TransitionReturn.FAILURE
            node.current_state = LifecycleState.TRANSITION_CONFIGURING
            success = node.on_configure_cb() if node.on_configure_cb else True
            if success:
                node.current_state = LifecycleState.PRIMARY_INACTIVE
                node.transition_history.append(f"{start_state.value}->INACTIVE")
                return TransitionReturn.SUCCESS
            else:
                node.current_state = LifecycleState.PRIMARY_UNCONFIGURED
                return TransitionReturn.FAILURE

        elif target_transition == "activate":
            if start_state != LifecycleState.PRIMARY_INACTIVE:
                logger.warning(f"{node_name}: Cannot activate from {start_state.value}")
                return TransitionReturn.FAILURE
            node.current_state = LifecycleState.TRANSITION_ACTIVATING
            success = node.on_activate_cb() if node.on_activate_cb else True
            if success:
                node.current_state = LifecycleState.PRIMARY_ACTIVE
                node.transition_history.append(f"{start_state.value}->ACTIVE")
                return TransitionReturn.SUCCESS
            else:
                # Rollback to INACTIVE
                node.current_state = LifecycleState.PRIMARY_INACTIVE
                return TransitionReturn.FAILURE

        elif target_transition == "deactivate":
            if start_state != LifecycleState.PRIMARY_ACTIVE:
                logger.warning(f"{node_name}: Cannot deactivate from {start_state.value}")
                return TransitionReturn.FAILURE
            node.current_state = LifecycleState.TRANSITION_DEACTIVATING
            success = node.on_deactivate_cb() if node.on_deactivate_cb else True
            node.current_state = LifecycleState.PRIMARY_INACTIVE
            node.transition_history.append(f"{start_state.value}->INACTIVE")
            return TransitionReturn.SUCCESS if success else TransitionReturn.FAILURE

        elif target_transition == "cleanup":
            if start_state != LifecycleState.PRIMARY_INACTIVE:
                logger.warning(f"{node_name}: Cannot cleanup from {start_state.value}")
                return TransitionReturn.FAILURE
            node.current_state = LifecycleState.TRANSITION_CLEANINGUP
            success = node.on_cleanup_cb() if node.on_cleanup_cb else True
            node.current_state = LifecycleState.PRIMARY_UNCONFIGURED
            node.transition_history.append(f"{start_state.value}->UNCONFIGURED")
            return TransitionReturn.SUCCESS if success else TransitionReturn.FAILURE

        elif target_transition == "shutdown":
            node.current_state = LifecycleState.TRANSITION_SHUTTINGDOWN
            if node.on_shutdown_cb:
                node.on_shutdown_cb()
            node.current_state = LifecycleState.PRIMARY_FINALIZED
            node.transition_history.append(f"{start_state.value}->FINALIZED")
            return TransitionReturn.SUCCESS

        return TransitionReturn.ERROR

    def configure_all(self) -> bool:
        """Transitions all registered nodes from UNCONFIGURED -> INACTIVE."""
        all_ok = True
        for name in self._managed_nodes:
            res = self.trigger_transition(name, "configure")
            if res != TransitionReturn.SUCCESS:
                all_ok = False
        return all_ok

    def activate_all(self) -> bool:
        """Transitions all registered nodes from INACTIVE -> ACTIVE."""
        all_ok = True
        for name in self._managed_nodes:
            res = self.trigger_transition(name, "activate")
            if res != TransitionReturn.SUCCESS:
                all_ok = False
        return all_ok

    def get_system_lifecycle_status(self) -> Dict[str, str]:
        """Returns current lifecycle state across all managed nodes."""
        return {name: node.current_state.value for name, node in self._managed_nodes.items()}


if __name__ == "__main__":
    mgr = ROS2LifecycleManager()
    mgr.register_node("perception_engine", on_configure=lambda: True, on_activate=lambda: True)
    mgr.register_node("diffusion_vla_policy", on_configure=lambda: True, on_activate=lambda: True)

    assert mgr.configure_all()
    assert mgr.activate_all()
    status = mgr.get_system_lifecycle_status()
    print("Lifecycle Status:", status)
    assert all(s == "ACTIVE" for s in status.values())
    print("Lifecycle Manager self-test passed.")

