from ..database import db

class BaseService:
    """
    Base class for all services.
    Provides access to the database connection.
    """
    def __init__(self):
        self.db = db

    def get_db(self):
        """Get the database manager singleton."""
        return self.db
