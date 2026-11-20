"""E-mail delivery. Dry-run (default) writes RFC-822 .eml files to the outbox. Real delivery, credentials from environment variables only:
  - SMTP with STARTTLS (e.g. Gmail: smtp.gmail.com:587 with a Google App Password), or
  - the Resend HTTPS API (HSM_RESEND_API_KEY), for hosts that block SMTP ports (Render free plan).
Retries with exponential backoff; every attempt's status is returned."""
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
        self.resend_key = os.environ.get(cfg.get('resend', {}).get('api_key_env', 'HSM_RESEND_API_KEY'))
        self.resend_from = os.environ.get('HSM_RESEND_FROM', 'Sensor Monitoring <onboarding@resend.dev>')

    def build(self, to, subject, text, html=None):
        m = EmailMessage(); m['From'] = self.sender; m['To'] = ', '.join(to); m['Subject'] = subject
        m['Message-ID'] = f'<{uuid.uuid4()}@hsm>'; m.set_content(text)
        if html: m.add_alternative(html, subtype='html')
        return m

    def send(self, to, subject, text, html=None, tag='msg'):
        m = self.build(to, subject, text, html)
        if self.dry:
            p = self.outbox / f'{tag}.eml'; p.write_bytes(bytes(m)); return dict(status='dry_run', attempts=0, path=str(p), error=None)
        if self.resend_key: return self._send_resend(to, subject, text, html)
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

    def _send_resend(self, to, subject, text, html):
        import json, urllib.request, urllib.error
        body = json.dumps(dict(**{'from': self.resend_from}, to=list(to), subject=subject, text=text, html=html or None)).encode()
        err = None
        for a in range(1, self.max_attempts + 1):
            try:
                req = urllib.request.Request('https://api.resend.com/emails', data=body, method='POST',
                                             headers={'Authorization': f'Bearer {self.resend_key}', 'Content-Type': 'application/json', 'User-Agent': 'hsm-monitor'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if 200 <= r.status < 300: return dict(status='sent', attempts=a, path=None, error=None)
                    err = f'HTTP {r.status}'
            except urllib.error.HTTPError as e:
                err = f'HTTP {e.code}: {e.read()[:300].decode("utf-8", "replace")}'
                if 400 <= e.code < 500 and e.code != 429: break          # bad key / address: retrying will not help
            except Exception as e: err = f'{type(e).__name__}: {e}'
            if a < self.max_attempts: self.sleep(2 ** a)
        return dict(status='failed', attempts=a, path=None, error=err)
