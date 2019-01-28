def pytest_addoption(parser):
    parser.addoption("--retrain", action="store_true", default=False)
    parser.addoption("--render", action="store_true", default=False)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.retrain
    if 'retrain' in metafunc.fixturenames:
        metafunc.parametrize("retrain", [option_value])