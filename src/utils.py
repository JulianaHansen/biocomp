import logging
import os

def config_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=os.path.join('logs', 'biocel.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )