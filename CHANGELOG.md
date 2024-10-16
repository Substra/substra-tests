# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [1.0.0](https://github.com/Substra/substra-tests/releases/tag/1.0.0) - 2024-10-14

### Added

- Add test checking function cancelation when linked compute plans are all canceled. ([#383](https://github.com/Substra/substra-tests/pull/383))

### Changed

- Upgraded scikit-learn and pandas test dependencies. ([#391](https://github.com/Substra/substra-tests/pull/391))
- Upgraded workflow dependencies. ([#393](https://github.com/Substra/substra-tests/pull/393))

### Removed

- Drop Python 3.9 support. ([#382](https://github.com/Substra/substra-tests/pull/382))


## [0.52.0](https://github.com/Substra/substra-tests/releases/tag/0.52.0) - 2024-09-12

### Added

- Python 3.12 support. ([#363](https://github.com/Substra/substra-tests/pull/363))

### Fixed

- Bump pytorch version to 2.2.1 in tests. ([#369](https://github.com/Substra/substra-tests/pull/369))

### Removed

- Dependency `pytest-lazy-fixture` ([#1509](https://github.com/Substra/substra-tests/pull/1509))


## [0.51.0](https://github.com/Substra/substra-tests/releases/tag/0.51.0) - 2024-06-03


### Changed

- Changed number of steps in `execution_rundown` ([#1517](https://github.com/Substra/substra-tests/pull/1517))


## [0.50.0](https://github.com/Substra/substra-tests/releases/tag/0.50.0) - 2024-03-27


No significant changes.


## [0.49.0] - 2024-03-07

### Changed

- Compute task status `DOING` is renamed `EXECUTING` ([#371](https://github.com/Substra/substra-tests/pull/371))
- Drop Python 3.8 support

## [0.48.0] - 2024-02-26

### Changed

- Nightly is now done on `owkin/substra-ci` repository ([#304](https://github.com/Substra/substra-tests/pull/304))
- Parallelism in SDK tests in deactivated until we fix the parallel compute plans issues ([#306](https://github.com/Substra/substra-tests/pull/306))
- A bunch of SDK tests are skipped due to regressions following the decoupled builder merge ([#306](https://github.com/Substra/substra-tests/pull/306))
- BREAKING: replace `todo_count` and `waiting_count` by the new counts following the new statuses in the backend ([#319](https://github.com/Substra/substra-tests/pull/319))
- Reactivated tests and parallelism in SDK tests ([#315](https://github.com/Substra/substra-tests/pull/315), [#317](https://github.com/Substra/substra-tests/pull/317))
- `wait_for_asset_synchronized` is now ignoring the `status` field, that can change over time ([#322](https://github.com/Substra/substra-tests/pull/322))

### Added

- Test case for function build on another backend than the task owner ([#318](https://github.com/Substra/substra-tests/pull/318))

## [0.47.0] - 2023-10-18

### Added

- Support on Python 3.11 ([#283](https://github.com/Substra/substra-tests/pull/283))

## [0.46.0] - 2023-10-02

### Changed

- Remove minimal and workflow substra-tools images. Only one substra-tools image is used ([#284](https://github.com/Substra/substra-tests/pull/284))

## [0.45.0] - 2023-09-08

### Changed

- Update to pydantic 2.3.0 ([#272](https://github.com/Substra/substra-tests/pull/272))

## [0.44.0] - 2023-09-07

### Changed

- Minor dependency updates. See commit history for more details.

## [0.43.0] - 2023-07-27

### Changed

- Remove `model` and `models` for input and output identifiers in tests. Replace by `shared` instead. ([#261](https://github.com/Substra/substra-tests/pull/261))
- Moved `wait_task` and `wait_compute_task` from `substratest.client.Client` to `substra.Client` ([#263](https://github.com/Substra/substra-tests/pull/263))

## [0.42.0] - 2023-06-27

### Changed

- Define test client (`substratest.client.Client`) as child class of `substra.Client` (#205)
  ([#257](https://github.com/Substra/substra-tests/pull/257))

## [0.41.0] - 2023-06-12

- Update tests using new substra sdk `list` and `get` functions ([#256](https://github.com/Substra/substra-tests/pull/256))

## [0.40.0] - 2023-05-11

### Added

- Test of a test task returning several performances ([#251](https://github.com/Substra/substra-tests/pull/251))

## [0.39.0] - 2023-03-31

### Changed

- BREAKING: rename Algo to Function (#243)
- Replace `task.outputs[identifier].value` by `client.get_task_output_asset(task.key, identifier).asset` (#256)

## [0.38.1] - 2023-02-06

### Added

- Remote test to verify that, with the right permission, we can test on an org from the training model of an other organization (#237)

### Changed

- chore: rename tuple to task (#239)

## [0.38.0] - 2023-01-30

### Changed

- Update test after test only for datasample removal (#235)

## [0.37.0] - 2022-12-19

### Added

- feat: add python 3.10 support (#232)
- Contributing, contributors & code of conduct files (#236)

## [0.36.0] - 2022-11-22

### Added

- Test case for `__` invalid in metadata key

### Changed

- Update code owners (#229)
- Update substratools Docker image (#230)
- Use the generic task (#221)
- Apply changes from algo to function in substratools (#224)
- Register functions in substratools using decorator `@tools.register` (#225)

## [0.35.0] - 2022-10-03

### Removed

- category field from algo as it's not required anymore

## [0.34.0] - 2022-09-26

- Update the Client, it takes a backend type instead of debug=True + env variable to set the spawner - (#210)

## [0.33.0] - 2022-09-19

### Removed

- Remove test compute plan remove intermediary model

### Fixed

- Algo inputs and task code consistency within test

## [0.32.0] - 2022-09-12

### Changed

- fix(tests): safe getter for optional inputs
- chore: Change the Dockerfile `ENTRYPOINT` definition of algorithm relying on `substratools` to pass the method to execute under the `--method-name` argument
- tests: Download tests optimized using the new substra feature: download function return the path to the downloaded file.

## [0.31.0] - 2022-08-29

### Removed

- Local folder tests

### Changed

- chore: Change the assets and tools algo to feed with inputs outputs dictionary.

### Added

- A test for task profiling reports

### Fixed

- Fix the test_base_substra_tools_image test.

## [0.30.0] - 2022-08-22

### Added

- Added a test for transient outputs

## [0.29.0] - 2022-08-17

### Fixed

- Fix the tests so that the task execution uses inputs/outputs

### Added

- Added tests for update name methods

## [0.28.0] - 2022-08-09

### Changed

- Drop Python 3.7 support

### Added

- Add expiration date to skipped tests

## [0.27.0] - 2022-08-01

### Changed

- BREAKING: Add "inputs" field to compute task specs

## [0.26.0] - 2022-07-11

### Changed

- BREAKING: Add "outputs" field to compute task specs

## [0.25.0] - 2022-07-11

### Changed

- BREAKING: convert (test task) to (predict task + test task)

## [0.24.0] - 2022-07-05

### Changed

- Stop using metrics APIs, use algo APIs instead
- Following the change in substra, rename `Client.update_compute_plan` in `Client.add_compute_plan_tuples`

## [0.23.0] - 2022-06-27

### Changed

- feat: use new filters implementation for list functions
- remove `test-ci` command, the ci run only the remote tests

## [0.22.0] - 2022-06-14

### Added

- workflow: configurable number of datasamples to use for Mnist

### Changed

- feat: rename node to organization

## [0.21.0] - 2022-06-07

### Added

- wait for tuple error type

## [0.20.0] - 2022-05-30

### Added

- feat: add slack bot for sdk local tests (publish on #component_substra_sdk with the Substra-CI app)

### Changed

- feat: in subprocess mode, errors are now caught as `ExecutionError`

## [0.19.0] - 2022-05-23

### Added

- Test if the GPU is seen from the algo

## [0.18.0] - 2022-05-16

### Changed

- BREAKING CHANGE: add mandatory name field to compute plan

## [0.17.0] - 2022-05-09

### Added

- BREAKING CHANGE: feat!: update tests to give a key to ComputePlanSpec
- Add a test on GPU support, skipped by default

## [0.16.0] - 2022-05-03

### Added

- test: get_performances from a compute plan key
- test: test the base docker image with kaniko #245

### Changed

- chore: refactor config

### Fixed

- fix: add missing type hint import

## [0.15.0] - 2022-04-19

### Changed

- feat: fully covering debug mode test

### Fixed

- fix: set remote tag to latest
- fix: permissions head model test

## [0.14.0] - 2022-04-11

### Added

- ci: option to chose orchestrator mode

### Changed

- ci: metric endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls
- ci: algo endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls
- ci: datamanager endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls
- ci: tuple endpoints now returns 404 instead of 400 for non-uuid keys in retrieve calls
- ci: compute plan endpoint now returns 404 instead of 400 for non-uuid keys in retrieve calls
- feat: use latest substra-tools image by default

## [0.13.0] - 2022-03-01

### Added

- feat: add support for substra GitHub Bot
- ci: workflow dispatch - be able to deactivate the sdk tests
- chore(ci): enable/disable frontend/substrafl tests when triggering the workflow manually

### Changed

- ci: the main branches of the substra and substra-tools repositories are now main
- The base Docker image is substra-tools 0.10.0, with Python 3.9

## [0.12.0] - 2022-01-16

### Added

- implement the substrafl e2e tests
- ci: export events logs
- add logs permission to dataset
- ci: Be able to run only the substrafl tests (#165 #173 #179)

### Changed

- increase coverage of algo.list view
- ci: the main branches of the substra and substra-tools repositories are now main

### Fixed

- ci: fix links in slack messages

## [0.11.0] - 2022-01-10

### Added

- add frontend testing, improve slack reporting
- test error type in tuple api responses

### Changed

- algo writes in the home directory
- format code with Black and isort
- update acceptable concommit subject pattern
- more compact Slack reports + add e2e tests duration

### Fixed

- GHA workflow slack reporting condition
- handle absent test duration in slack reporting

## [0.10.0] - 2021-12-01

### Added

- test on algo download

### Changed

- update tests for compatibility with backend
- improve tests coverage on task compatibility

## [0.9.0] - 2021-11-02

### Added

- test returned task counts with cp
- add default values for user inputs
- test start end date integration
- use kaniko prepolutaed images
- add echo step of workflow inputs
- make repo branches configurable for e2e tests in github action
- enforce conventional PR titles
- multiple metrics per testtask

### Changed

- rename tests.test_debug module
- improve code for local images
- update skaffold version
- update the webhook to target a new channel
- Retry a bit when images fail building

### Fixed

- handle canceled and failed cp in client.wait method
- task registration on a disabled model should fail
- remove the random id generator that fails everyday
- remove a pipe to avoid a broken pipe error
- missing inputs field
- fix workflow indentation
- add helm couchdb repo
- do not fail if output adds extra lines
- update metric and testtuple add for the mnist workflow
- access the local folder path properly instead of using a hardcoded path

## [0.8.0] - 2021-10-04

### Added

- Option to `run-ci.py` to create a cluster with multiple kubernetes nodes
- Options `--no-cleanup` and `--use-cluster` to `run-ci.py`
- `cleanup.py` in addition to `run-ci.py`
- Orchestrator
- Run nightly tests in standalone and distributed mode
- Twuni helm repo

### Changed

- Unify all algos into one
- Testtuple after aggregate tuple

### Removed

### Fixed

## [0.7.0] - 2021-08-04

## [0.6.2] - 2021-06-28

## [0.6.1] - 2021-04-13
