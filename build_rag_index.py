"""
Build the FAISS RAG index. Run once before launching the app.

Usage:
    python build_rag_index.py                          # auto-detect paths
    python build_rag_index.py --max 50                 # quick test (50 images)
    python build_rag_index.py --force                  # force rebuild
    python build_rag_index.py --qa ./dme_vqa/qa/trainqa.json --images ./dme_vqa/visual/train
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="DeepEye RAG Index Builder")
    parser.add_argument("--max",    type=int,  default=9428,  help="Max records to index")
    parser.add_argument("--force",  action="store_true",       help="Rebuild even if exists")
    parser.add_argument("--qa",     type=str,  default="",     help="Path to trainqa.json")
    parser.add_argument("--images", type=str,  default="",     help="Path to image folder")
    args = parser.parse_args()

    print("="*55)
    print("  DeepEye — RAG Index Builder")
    print("="*55)
    print(f"  Working dir  : {os.getcwd()}")
    print(f"  Max records  : {args.max}")
    print(f"  Force rebuild: {args.force}")
    print("="*55)

    os.system("pip install faiss-cpu -q")

    import rag_engine

    # Override paths if provided via CLI
    if args.qa:
        rag_engine.QA_FILE = args.qa
        print(f"  QA file (override): {args.qa}")
    if args.images:
        rag_engine.IMAGE_DIR = args.images
        print(f"  Image dir (override): {args.images}")

    from rag_engine import build_index, index_status

    status = index_status()
    if status["exists"] and not args.force:
        print(f"\nIndex already exists: {status['n_total']} entries ({status['size_mb']} MB)")
        print("Use --force to rebuild.")
        return

    try:
        build_index(force=args.force, max_records=args.max)
    except FileNotFoundError as e:
        print(e)
        print("\n── How to fix ───────────────────────────────")
        print("  Option A: Run from the folder containing dme_vqa/:")
        print("    cd C:\\Users\\ISHHAQ\\...\\DeepEyeNet")
        print("    python deepeye_app/build_rag_index.py")
        print()
        print("  Option B: Pass paths explicitly:")
        print("    python build_rag_index.py \\")
        print("      --qa   C:/path/to/dme_vqa/qa/trainqa.json \\")
        print("      --images C:/path/to/dme_vqa/visual/train")
        sys.exit(1)

    status = index_status()
    print(f"\n✅ Done! {status['n_total']} entries · {status['size_mb']} MB")
    print("Now run:  streamlit run app.py")

if __name__ == "__main__":
    main()
