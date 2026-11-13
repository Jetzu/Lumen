"""Drop-in API routers.

Every module in this package that defines ``router = APIRouter()`` is mounted under ``/api``
automatically by ``lumen.server.app.create_app``.  Routers receive the session via
``from ..app import get_session`` → ``get_session()``.
"""
