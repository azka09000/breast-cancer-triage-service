"""Optional Redis-backed counter of served predictions."""
import logging

import redis

logger = logging.getLogger("triage.counter")
COUNTER_KEY = "predictions_served"


class PredictionCounter:
    """Counts predictions in Redis when REDIS_HOST is set; never breaks a prediction."""

    def __init__(self, host, port):
        self.enabled = host is not None
        self._client = None
        if self.enabled:
            self._client = redis.Redis(
                host=host, port=port, decode_responses=True,
                socket_connect_timeout=1, socket_timeout=1,
            )

    def increment(self):
        if not self.enabled:
            return None
        try:
            return int(self._client.incr(COUNTER_KEY))
        except redis.RedisError as exc:
            logger.warning("Could not increment the prediction counter: %s", exc)
            return None

    def current(self):
        if not self.enabled:
            return None
        try:
            value = self._client.get(COUNTER_KEY)
            return int(value) if value is not None else 0
        except redis.RedisError as exc:
            logger.warning("Could not read the prediction counter: %s", exc)
            return None
