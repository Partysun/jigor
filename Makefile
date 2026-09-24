# jigor: build, test and serve entry point.
#
#   make build            # debug build (lib + jigor binary + examples)
#   make test             # cargo unit tests + hurl integration tests
#   make unit             # cargo unit tests only
#   make integration      # hurl integration tests (spins up its own server)
#   make serve            # run the HTTP gateway (jigor serve, default :8000)
#   make dev              # run the decide example
#   make bench            # run the release benchmark example
#   make lint             # cargo clippy
#   make fmt              # cargo fmt (check mode: make fmt-check)

.DEFAULT_GOAL := build

# Server host/port (also honoured by the hurl runner).
JIGOR_HOST  ?= 127.0.0.1
JIGOR_PORT  ?= 8000
JIGOR_DEVICE?=
export JIGOR_DEVICE

.PHONY: build release unit integration cli test serve install publish dev tweet tweet-test tagger tagger-test bench lint fmt fmt-check help

build:
	cargo build

release:
	cargo build --release

unit:
	cargo test

integration:
	bash tests/hurl/run.sh

cli:
	bash tests/cli/ask.sh

test: unit integration cli

serve:
	cargo build -p jigor-cli
	./target/debug/jigor serve --host "$(JIGOR_HOST)" --port "$(JIGOR_PORT)"

install:
	cargo install --path ./crates/jigor-cli --locked

publish:
	cargo publish -p jigor
	cargo publish -p jigor-cli

dev:
	cargo run -p jigor --example decide

tweet:
	cargo run -p jigor --example tweet -- "We just crossed 10,000 paying customers. Thank you."

tweet-test:
	cargo test -p jigor --example tweet

tagger:
	cargo run -p jigor --example tagger -- --title "Hiring notes" --tags "work, ideas, personal" "Talked through the Q4 roadmap and the hiring push. Budget approved for two engineers."

tagger-test:
	cargo test -p jigor --example tagger

bench:
	cargo run -p jigor --release --example bench

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

help:
	@printf '%s\n' \
		'jigor targets:' \
		'  build        debug build (lib + jigor binary + examples)' \
		'  release      release build of the jigor binary' \
		'  test         unit tests + hurl integration tests' \
		'  unit         cargo unit tests' \
		'  integration  hurl integration tests (spins up its own server)' \
		'  serve        run the HTTP server (JIGOR_HOST/JIGOR_PORT overridable)' \
		'  dev          run the decide example' \
		'  bench        run the release benchmark' \
		'  lint         cargo clippy --all-targets -D warnings' \
		'  fmt          cargo fmt | fmt-check for check mode'