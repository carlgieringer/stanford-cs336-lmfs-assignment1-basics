"""
Lambda Cloud API integration for instance management.
"""

import getpass
import json
import logging
import socket
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class LambdaCloudError(Exception):
    """Exception raised for Lambda Cloud API errors."""
    pass


def get_local_ip() -> str:
    """Get the local IP address of the current machine."""
    try:
        # Connect to a remote address to determine local IP
        # This doesn't actually send data, just determines routing
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception as e:
        raise LambdaCloudError(f"Failed to get local IP address: {e}")


def get_api_key() -> str:
    """Prompt user for Lambda Cloud API key securely."""
    api_key = getpass.getpass("Enter your Lambda Cloud API key: ")
    if not api_key.strip():
        raise LambdaCloudError("API key cannot be empty")
    return api_key.strip()


def list_instances(api_key: str) -> Dict[str, Any]:
    """List all Lambda Cloud instances."""
    url = "https://cloud.lambdalabs.com/api/v1/instances"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise LambdaCloudError(f"Failed to list instances: {e}")


def find_instance_by_ip(instances_data: Dict[str, Any], target_ip: str) -> Optional[str]:
    """Find instance ID by IP address."""
    if "data" not in instances_data:
        raise LambdaCloudError("Invalid instances response format")

    for instance in instances_data["data"]:
        if instance.get("ip") == target_ip:
            return instance.get("id")

    return None


def terminate_instance(api_key: str, instance_id: str) -> None:
    """Terminate a Lambda Cloud instance."""

    url = "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "instance_ids": [instance_id]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        logger.info(f"Successfully initiated termination for instance {instance_id}")
    except requests.exceptions.RequestException as e:
        raise LambdaCloudError(f"Failed to terminate instance {instance_id}: {e}")


def validate_and_get_instance_info(api_key: str) -> tuple[str, str]:
    """
    Validate that the current host is a Lambda Cloud instance and return instance info.

    Returns:
        tuple: (local_ip, instance_id)

    Raises:
        LambdaCloudError: If validation fails or instance not found
    """
    logger.info("Validating Lambda Cloud instance...")

    # Get local IP
    local_ip = get_local_ip()
    logger.info(f"Local IP address: {local_ip}")

    # List instances from Lambda Cloud
    instances_data = list_instances(api_key)
    logger.info(f"Retrieved {len(instances_data.get('data', []))} instances from Lambda Cloud")

    # Find current instance
    instance_id = find_instance_by_ip(instances_data, local_ip)
    if not instance_id:
        available_ips = [inst.get("ip") for inst in instances_data.get("data", [])]
        raise LambdaCloudError(
            f"Current host IP {local_ip} not found in Lambda Cloud instances. "
            f"Available instance IPs: {available_ips}. "
            f"This suggests you're not running on a Lambda Cloud instance or the --terminate-at-end flag was used incorrectly."
        )

    logger.info(f"Found current instance: {instance_id}")
    return local_ip, instance_id


def terminate_current_instance(api_key) -> None:
    """
    Terminate the current Lambda Cloud instance.

    Args:
        api_key: Lambda Cloud API key. If None, will prompt for it.

    This function:
    1. Gets API key (prompts if not provided)
    2. Validates the current host is a Lambda Cloud instance
    3. Terminates the instance
    """
    logger.info("Starting Lambda Cloud instance termination process...")

    local_ip, instance_id = validate_and_get_instance_info(api_key)

    logger.info(f"Terminating Lambda Cloud instance {instance_id} (IP: {local_ip})...")
    terminate_instance(api_key, instance_id)

    logger.info("Instance termination initiated successfully. The instance will shut down shortly.")
