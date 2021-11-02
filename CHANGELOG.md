# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2021-11-02

### Added
- test returned task counts with cp (#91)
- add default values for user inputs (#99)
- test start end date integration (#93)
- use kaniko prepolutaed images (#95)
- add echo step of workflow inputs (#90)
- make repo branches configurable for e2e tests in github action (#86)
- enforce conventional PR titles (#84)
- multiple metrics per testtask (#74)

### Changed
- rename tests.test_debug module (#103)
- improve code for local images (#102)
- update skaffold version (#80)
- update the webhook to target a new channel (#81)
- Retry a bit when images fail building (#79)

### Fixed
- handle canceled and failed cp in client.wait method (#100)
- task registration on a disabled model should fail (#97)
- remove the random id generator that fails everyday (#98)
- remove a pipe to avoid a broken pipe error (#94)
- missing inputs field (#89)
- fix workflow indentation (#88)
- add helm couchdb repo (#85)
- do not fail if output adds extra lines (#83)
- update metric and testtuple add for the mnist workflow (#82)
- access the local folder path properly instead of using a hardcoded path (#76)

## [0.8.0] - 2021-10-04

### Added
- Option to `run-ci.py` to create a cluster with multiple kubernetes nodes (#37)
- Options `--no-cleanup` and `--use-cluster` to `run-ci.py` (#39)
- `cleanup.py` in addition to `run-ci.py` (#49)
- Orchestrator (#2)
- Run nightly tests in standalone and distributed mode (#63)
- Twuni helm repo (#65)

### Changed
- Unify all algos into one (#61)
- Testtuple after aggregate tuple (#32)

### Removed

### Fixed

## [0.7.0] - 2021-08-04

## [0.6.2] - 2021-06-28

## [0.6.1] - 2021-04-13
