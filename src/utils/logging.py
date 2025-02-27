def setup_log():
    import logging
    import os
    from datetime import datetime

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Set up logging configuration
    log_file = os.path.join('logs', f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger(__name__)