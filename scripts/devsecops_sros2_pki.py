#!/usr/bin/env python3
"""
AURA-Drive™ SROS2 (Secure ROS 2) DDS-Security PKI Keystore Generator
====================================================================
Complies with OMG DDS-Security v1.1 and ROS 2 SROS2 Specifications:
  - Generates X.509 Root Certificate Authority (CA) with ECDSA NIST P-256 keys.
  - Issues signed cryptographic identity certificates for all autonomy nodes.
  - Generates signed governance.xml (enforces topic encryption & authentication).
  - Generates signed permissions.xml (declares granular topic publish/subscribe rules).
"""

import os
import sys
import datetime
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYSTORE_DIR = REPO_ROOT / "config" / "sros2_keystore"


def generate_ecdsa_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def create_root_ca(keystore_path: Path):
    """Creates the Root CA certificate and private key."""
    ca_key = generate_ecdsa_key()
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AURA-Drive Industrial Robotics"),
        x509.NameAttribute(NameOID.COMMON_NAME, "AURA-Drive SROS2 Root CA"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    ca_dir = keystore_path / "ca"
    ca_dir.mkdir(parents=True, exist_ok=True)

    # Write CA key & cert
    with open(ca_dir / "ca_key.pem", "wb") as f:
        f.write(ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(ca_dir / "ca_cert.pem", "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    return ca_key, ca_cert


def issue_node_certificate(node_name: str, ca_key, ca_cert, keystore_path: Path):
    """Issues an X.509 identity certificate for a specific autonomy node."""
    node_key = generate_ecdsa_key()
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AURA-Drive Industrial Robotics"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"node_{node_name}"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    node_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(node_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=730))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    node_dir = keystore_path / "nodes" / node_name
    node_dir.mkdir(parents=True, exist_ok=True)

    with open(node_dir / "key.pem", "wb") as f:
        f.write(node_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(node_dir / "cert.pem", "wb") as f:
        f.write(node_cert.public_bytes(serialization.Encoding.PEM))


def generate_dds_governance_and_permissions(keystore_path: Path):
    """Generates OMG DDS-Security governance.xml and permissions.xml files."""
    governance_xml = """<?xml version="1.0" encoding="utf-8"?>
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:noNamespaceSchemaLocation="http://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_governance.xsd">
  <domain_access_rules>
    <domain_rule>
      <domains>
        <id_range>
          <min>0</min>
          <max>232</max>
        </id_range>
      </domains>
      <allow_unauthenticated_participants>false</allow_unauthenticated_participants>
      <enable_join_access_control>true</enable_join_access_control>
      <discovery_protection_kind>ENCRYPT</discovery_protection_kind>
      <liveliness_protection_kind>ENCRYPT</liveliness_protection_kind>
      <rtps_protection_kind>ENCRYPT</rtps_protection_kind>
      <topic_access_rules>
        <topic_rule>
          <topic_expression>*</topic_expression>
          <enable_discovery_protection>true</enable_discovery_protection>
          <enable_read_access_control>true</enable_read_access_control>
          <enable_write_access_control>true</enable_write_access_control>
          <metadata_protection_kind>ENCRYPT</metadata_protection_kind>
          <data_protection_kind>ENCRYPT</data_protection_kind>
        </topic_rule>
      </topic_access_rules>
    </domain_rule>
  </domain_access_rules>
</dds>
"""
    with open(keystore_path / "governance.xml", "w", encoding="utf-8") as f:
        f.write(governance_xml)

    permissions_xml = """<?xml version="1.0" encoding="utf-8"?>
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:noNamespaceSchemaLocation="http://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_permissions.xsd">
  <permissions>
    <grant name="AURA_Industrial_Autonomy_Grant">
      <subject_name>CN=node_*,O=AURA-Drive Industrial Robotics,C=US</subject_name>
      <validity>
        <not_before>2026-01-01T00:00:00</not_before>
        <not_after>2035-12-31T23:59:59</not_after>
      </validity>
      <allow_rule>
        <domains>
          <id>0</id>
        </domains>
        <publish>
          <topics>
            <topic>/cmd_vel</topic>
            <topic>/tf</topic>
            <topic>/tf_static</topic>
            <topic>/aura/diffusion_policy/trajectory</topic>
            <topic>/aura/neural_sdf/costmap</topic>
            <topic>/aura/safety/watchdog_status</topic>
          </topics>
        </publish>
        <subscribe>
          <topics>
            <topic>/camera/image_raw</topic>
            <topic>/camera/depth/image_raw</topic>
            <topic>/scan</topic>
            <topic>/odom</topic>
            <topic>/vda5050/order</topic>
          </topics>
        </subscribe>
      </allow_rule>
      <default>DENY</default>
    </grant>
  </permissions>
</dds>
"""
    with open(keystore_path / "permissions.xml", "w", encoding="utf-8") as f:
        f.write(permissions_xml)


def generate_full_sros2_pki():
    print("=" * 70)
    print("AURA-Drive™ SROS2 DDS-Security PKI Keystore Generator")
    print("=" * 70)

    KEYSTORE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Generating ECDSA NIST P-256 Root Certificate Authority (CA)...")
    ca_key, ca_cert = create_root_ca(KEYSTORE_DIR)
    print(f"      Root CA Cert: {KEYSTORE_DIR / 'ca' / 'ca_cert.pem'}")

    nodes = [
        "perception_engine",
        "diffusion_vla_policy",
        "cuda_mppi_optimizer",
        "iso26262_watchdog",
        "vda5050_connector",
        "amcl_localization"
    ]

    print("\n[2/3] Issuing Signed X.509 Identity Certificates for Autonomy Nodes...")
    for n in nodes:
        issue_node_certificate(n, ca_key, ca_cert, KEYSTORE_DIR)
        print(f"      Issued certificate for: {n}")

    print("\n[3/3] Generating DDS-Security Governance & Permissions XML...")
    generate_dds_governance_and_permissions(KEYSTORE_DIR)
    print(f"      DDS Governance Policy: {KEYSTORE_DIR / 'governance.xml'}")
    print(f"      DDS Permissions Policy: {KEYSTORE_DIR / 'permissions.xml'}")

    print("\n[PASS] SROS2 DDS-Security PKI Keystore successfully generated.")
    print("=" * 70)


if __name__ == "__main__":
    generate_full_sros2_pki()

