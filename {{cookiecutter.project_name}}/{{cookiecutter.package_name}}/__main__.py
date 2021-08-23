def main():
    package = __package__ or "commands"
    app = __import__(package).app
    app()


if __name__ == "__main__":
    main()
