-- Beta 6 UI Refactor: Ensure corrected_by column exists for audit trail
ALTER TABLE correction_history ADD COLUMN corrected_by TEXT DEFAULT 'user';
