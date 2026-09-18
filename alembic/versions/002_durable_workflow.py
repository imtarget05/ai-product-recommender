"""002_durable_workflow

Revision ID: 002_durable_workflow
Revises: 001_initial_schema
Create Date: 2026-09-18 17:30:00.000000

Shared durable-ops tables (Plan 02 contract): idempotency, jobs, outbox,
dead letters, audit events, model registry. Portable column types so the
same migration runs on SQLite (tests/dev) and Postgres (Neon staging).

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_durable_workflow'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Idempotency keys (composite caller scope + key)
    op.create_table(
        'idempotency_keys',
        sa.Column('caller_scope', sa.String(length=128), nullable=False),
        sa.Column('idem_key', sa.String(length=128), nullable=False),
        sa.Column('fingerprint', sa.Text(), nullable=False, server_default=''),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('response', sa.Text(), nullable=False, server_default=''),
        sa.Column('expires_at', sa.String(length=64), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('caller_scope', 'idem_key')
    )

    # 2. Durable async jobs
    op.create_table(
        'jobs',
        sa.Column('job_id', sa.String(length=64), nullable=False),
        sa.Column('kind', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='queued'),
        sa.Column('attempt', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('lease_expires_at', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('input_ref', sa.Text(), nullable=False, server_default=''),
        sa.Column('result_ref', sa.Text(), nullable=False, server_default=''),
        sa.Column('error_class', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('updated_at', sa.String(length=64), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index(op.f('ix_jobs_status'), 'jobs', ['status'], unique=False)

    # 3. Reliable delivery outbox
    op.create_table(
        'outbox_events',
        sa.Column('event_id', sa.String(length=64), nullable=False),
        sa.Column('destination', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('payload', sa.Text(), nullable=False, server_default=''),
        sa.Column('version', sa.String(length=32), nullable=False, server_default='v1'),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('next_retry_at', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('delivered_at', sa.String(length=64), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('event_id')
    )
    op.create_index(op.f('ix_outbox_next_retry'), 'outbox_events', ['next_retry_at'], unique=False)

    # 4. Dead letters awaiting operator replay decision
    op.create_table(
        'dead_letters',
        sa.Column('job_id', sa.String(length=64), nullable=False),
        sa.Column('input_ref', sa.Text(), nullable=False, server_default=''),
        sa.Column('diagnosis', sa.Text(), nullable=False, server_default=''),
        sa.Column('owner', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('replay_decision', sa.String(length=32), nullable=False, server_default='pending'),
        sa.PrimaryKeyConstraint('job_id')
    )

    # 5. Generic audit trail
    op.create_table(
        'audit_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('actor', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('action', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('object', sa.String(length=256), nullable=False, server_default=''),
        sa.Column('request_id', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('outcome', sa.String(length=32), nullable=False, server_default=''),
        sa.Column('reason', sa.Text(), nullable=False, server_default=''),
        sa.Column('created_at', sa.String(length=64), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_request_id'), 'audit_events', ['request_id'], unique=False)

    # 6. Promoted model bundle versions
    op.create_table(
        'model_registry',
        sa.Column('version', sa.String(length=64), nullable=False),
        sa.Column('fingerprint', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('corpus_version', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='staged'),
        sa.Column('promoted_at', sa.String(length=64), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('version')
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_request_id'), table_name='audit_events')
    op.drop_table('model_registry')
    op.drop_table('audit_events')
    op.drop_table('dead_letters')
    op.drop_index(op.f('ix_outbox_next_retry'), table_name='outbox_events')
    op.drop_table('outbox_events')
    op.drop_index(op.f('ix_jobs_status'), table_name='jobs')
    op.drop_table('jobs')
    op.drop_table('idempotency_keys')
