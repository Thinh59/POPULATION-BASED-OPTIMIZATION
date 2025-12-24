import sys
from pathlib import Path


src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

menu_dir = Path(__file__).parent / 'menu'
if str(menu_dir) not in sys.path:
    sys.path.insert(0, str(menu_dir))

from menu.continuous_mode import continuous_menu

def main():
    continuous_menu()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user.")
        print(" Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)