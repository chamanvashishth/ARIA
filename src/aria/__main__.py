from aria.core.contracts import SystemStatus


def main() -> None:
    status = SystemStatus(
        name="ARIA",
        version="0.1.0",
        phase="foundation",
        implemented=False,
    )
    print(f"{status.name} {status.version} — {status.phase} phase")
    print("Core AI capabilities are not implemented yet.")


if __name__ == "__main__":
    main()
