
import sys
import os
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from utils.segment_utils import validate_activity_for_room

def test_validation():
    rooms = ['livingroom', 'bedroom', 'bathroom', 'kitchen', 'entrance']
    
    print("Testing 'unoccupied' validation for all rooms...")
    
    all_passed = True
    for room in rooms:
        result = validate_activity_for_room('unoccupied', room)
        if result == 'unoccupied':
            print(f"  [PASS] {room}: 'unoccupied' -> '{result}'")
        else:
            print(f"  [FAIL] {room}: 'unoccupied' -> '{result}' (Expected 'unoccupied')")
            all_passed = False
            
    if all_passed:
        print("\nSUCCESS: 'unoccupied' is valid for all rooms.")
    else:
        print("\nFAILURE: specific rooms failed validation.")
    assert all_passed


def test_unknown_validation():
    rooms = ['livingroom', 'bedroom', 'bathroom', 'kitchen', 'entrance']
    for room in rooms:
        result = validate_activity_for_room('unknown', room)
        assert result == 'unknown'

if __name__ == "__main__":
    test_validation()
