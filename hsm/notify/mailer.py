"""E-mail delivery. Dry-run (default) writes RFC-822 .eml files to the outbox; real delivery uses SMTP with STARTTLS and
credentials from environment variables only. Retries with exponential backoff; every attempt's status is returned."""
import os, smtplib, ssl, time, uuid
from email.message import EmailMessage
from pathlib import Path

class Mailer:
    def __init__(self, cfg, sleep=time.sleep):
        self.cfg = cfg; self.sleep = sleep
        self.dry = (not cfg.get('enabled', False)) or cfg.get('dry_run', True)
        self.outbox = Path(cfg.get('outbox_dir', './outbox')); self.outbox.mkdir(parents=True, exist_ok=True)
        sm = cfg.get('smtp', {})
        self.host = os.environ.get(sm.get('host_env', 'HSM_SMTP_HOST')); self.port = int(os.environ.get(sm.get('port_env', 'HSM_SMTP_PORT'), 587))
        self.user = os.environ.get(sm.get('username_env', 'HSM_SMTP_USER')); self.password = os.environ.get(sm.get('password_env', 'HSM_SMTP_PASSWORD'))
        self.sender = os.environ.get(sm.get('from_env', 'HSM_SMTP_FROM'), 'hsm-monitor@localhost')
        self.starttls = sm.get('starttls', True); self.timeout = sm.get('timeout_s', 20); self.max_attempts = sm.get('max_attempts', 4)

    def build(self, to, subject, text, html=None):
        m = EmailMessage(); m['From'] = self.sender; m['To'] = ', '.join(to); m['Subject'] = subject
        m['Message-ID'] = f'<{uuid.uuid4()}@hsm>'; m.set_content(text)
        if html: m.add_alternative(html, subtype='html')
        return m

    def send(self, to, subject, text, html=None, tag='msg'):
        m = self.build(to, subject, text, html)
        if self.dry:
            p = self.outbox / f'{tag}.eml'; p.write_bytes(bytes(m)); return dict(status='dry_run', attempts=0, path=str(p), error=None)
        if not self.host:
            return dict(status='failed', attempts=0, path=None, error='SMTP host not configured (environment variable missing)')
        err = None
        for a in range(1, self.max_attempts + 1):
            try:
                with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
                    if self.starttls: s.starttls(context=ssl.create_default_context())
                    if self.user: s.login(self.user, self.password or '')
                    s.send_message(m)
                return dict(status='sent', attempts=a, path=None, error=None)
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
                if a < self.max_attempts: self.sleep(2 ** a)
        return dict(status='failed', attempts=self.max_attempts, path=None, error=err)
