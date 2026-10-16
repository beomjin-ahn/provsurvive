"""A local, TEST-ONLY certificate authority for signing C2PA manifests.

Chain: root CA -> intermediate CA -> leaf (ES256 / P-256). The leaf follows the
C2PA certificate profile: digitalSignature key usage, an explicit EKU
(emailProtection + documentSigning), not self-signed, 10-year validity so that
runs stay valid offline without a time-stamp authority.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

VALIDITY = dt.timedelta(days=3650)
_DOCUMENT_SIGNING = x509.ObjectIdentifier("1.3.6.1.5.5.7.3.36")
_ORG = "provsurvive TEST ONLY"


def _fingerprint(pem: bytes) -> str:
    return x509.load_pem_x509_certificate(pem).fingerprint(hashes.SHA256()).hex()


@dataclass(frozen=True)
class TrustAnchor:
    """The root certificate alone: enough to validate manifests, nothing to sign with."""

    root_pem: bytes

    @property
    def root_fingerprint(self) -> str:
        return _fingerprint(self.root_pem)


@dataclass(frozen=True)
class TestCA:
    root_pem: bytes
    chain_pem: bytes  # leaf + intermediate, as the signer presents it
    leaf_key_pem: bytes

    @property
    def root_fingerprint(self) -> str:
        return _fingerprint(self.root_pem)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "root.pem").write_bytes(self.root_pem)
        (directory / "chain.pem").write_bytes(self.chain_pem)
        key_path = directory / "leaf.key"
        key_path.write_bytes(self.leaf_key_pem)
        key_path.chmod(0o600)

    @classmethod
    def load(cls, directory: Path) -> TestCA:
        return cls(
            root_pem=(directory / "root.pem").read_bytes(),
            chain_pem=(directory / "chain.pem").read_bytes(),
            leaf_key_pem=(directory / "leaf.key").read_bytes(),
        )

    @classmethod
    def load_or_create(cls, directory: Path) -> TestCA:
        if (directory / "leaf.key").exists():
            return cls.load(directory)
        ca = create_test_ca()
        ca.save(directory)
        return ca


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, _ORG),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _cert(
    subject: x509.Name,
    key: ec.EllipticCurvePrivateKey,
    issuer: x509.Name,
    issuer_key: ec.EllipticCurvePrivateKey,
    now: dt.datetime,
    path_length: int | None,
) -> x509.Certificate:
    """A CA certificate, or with ``path_length`` None the C2PA signer (leaf) certificate."""
    ca = path_length is not None
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + VALIDITY)
        .add_extension(x509.BasicConstraints(ca=ca, path_length=path_length), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=not ca,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=ca,
                crl_sign=ca,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    )
    if not ca:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION, _DOCUMENT_SIGNING]),
            critical=False,
        )
    return (
        builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()),
            critical=False,
        )
        .sign(issuer_key, hashes.SHA256())
    )


def create_test_ca(now: dt.datetime | None = None) -> TestCA:
    now = now or dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    root_key = ec.generate_private_key(ec.SECP256R1())
    int_key = ec.generate_private_key(ec.SECP256R1())
    leaf_key = ec.generate_private_key(ec.SECP256R1())

    root_name = _name("provsurvive TEST Root CA")
    int_name = _name("provsurvive TEST Intermediate CA")
    leaf_name = _name("provsurvive TEST signer")
    root = _cert(root_name, root_key, root_name, root_key, now, path_length=1)
    inter = _cert(int_name, int_key, root_name, root_key, now, path_length=0)
    leaf = _cert(leaf_name, leaf_key, int_name, int_key, now, path_length=None)

    pem = serialization.Encoding.PEM
    return TestCA(
        root_pem=root.public_bytes(pem),
        chain_pem=leaf.public_bytes(pem) + inter.public_bytes(pem),
        leaf_key_pem=leaf_key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
    )
