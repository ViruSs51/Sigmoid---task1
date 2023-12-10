import asyncio
import warnings

from app import App

if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    config_path = 'data/config.json'

    while True:
        run_type = input('\n1) Scrieti "learn" pentru a invata inteligenta artificiala\n2) Scrieti lista de date separata prin viruga(exemplu: "1,24,3,4") pentru a utiliza AI\n>>> ')

        app = App(config_file=config_path, get_weighting=True)
        asyncio.run(app.run(type=run_type))