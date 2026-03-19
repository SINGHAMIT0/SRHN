##
## SRHN v4 Makefile — Production Build
## Targets: all, train, test, clean, release, debug, docker
##
CC      = gcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra \
           -I./include \
           -D_POSIX_C_SOURCE=200809L \
           -Wno-unused-parameter \
           -Wno-unused-function
LDFLAGS = -lpthread -lm -ldl

SRC_DIR   = src
TEST_DIR  = tests
BUILD_DIR = build

## Core library (v4 base)
LIB_SRCS = \
  $(SRC_DIR)/srhn_core4.c \
  $(SRC_DIR)/srhn_embed4.c \
  $(SRC_DIR)/srhn_tokenizer4.c \
  $(SRC_DIR)/srhn_hnsw4.c \
  $(SRC_DIR)/srhn_reasoning4.c \
  $(SRC_DIR)/srhn_llm4.c \
  $(SRC_DIR)/srhn_feedback4.c \
  $(SRC_DIR)/srhn_memory4.c \
  $(SRC_DIR)/srhn_persist4.c \
  $(SRC_DIR)/srhn_utils4.c \
  $(SRC_DIR)/srhn_api4.c

## v4 Research modules
RESEARCH_SRCS = \
  $(SRC_DIR)/srhn_multisem.c \
  $(SRC_DIR)/srhn_hyperedge_attn.c \
  $(SRC_DIR)/srhn_autoenc.c \
  $(SRC_DIR)/srhn_vae.c \
  $(SRC_DIR)/srhn_pacbayes_spectral.c \
  $(SRC_DIR)/srhn_scale.c

ALL_LIB_SRCS = $(LIB_SRCS) $(RESEARCH_SRCS)
LIB_OBJS     = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(ALL_LIB_SRCS))
TARGET       = $(BUILD_DIR)/libsrhn4.a

## Training module
TRAIN_SRCS = $(SRC_DIR)/srhn_v4_train.c
TRAIN_OBJS = $(BUILD_DIR)/srhn_v4_train.o
TRAIN_BIN  = $(BUILD_DIR)/srhn4_train

## Test binary
TEST_BIN   = $(BUILD_DIR)/srhn4_test

.PHONY: all train test clean dirs release debug docker

all: dirs $(TARGET) $(TRAIN_BIN) $(TEST_BIN)

train: dirs $(TARGET) $(TRAIN_BIN)

dirs:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(LIB_OBJS)
	ar rcs $@ $^
	@echo "[build] Library:  $@"

$(TRAIN_BIN): $(TRAIN_OBJS) $(BUILD_DIR)/srhn_v4_train_main.o $(TARGET)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "[build] Trainer:  $@"

$(BUILD_DIR)/srhn_v4_train.o: $(SRC_DIR)/srhn_v4_train.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/srhn_v4_train_main.o: $(SRC_DIR)/srhn_v4_train_main.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): $(LIB_OBJS) $(TEST_DIR)/srhn4_test.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "[build] Tests:    $@"

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)

## Debug — AddressSanitizer + UBSan
debug: CFLAGS += -g3 -fsanitize=address,undefined -DDEBUG -fno-omit-frame-pointer
debug: all

## Release — maximum optimisation
release: CFLAGS  += -O3 -march=native -flto -DNDEBUG
release: LDFLAGS += -flto
release: all

## Docker build
docker:
	docker build -t srhn4:latest .

## Module-level rebuild shortcuts
core:   dirs $(BUILD_DIR)/srhn_core4.o
ae:     dirs $(BUILD_DIR)/srhn_autoenc.o
vae:    dirs $(BUILD_DIR)/srhn_vae.o
spec:   dirs $(BUILD_DIR)/srhn_pacbayes_spectral.o
multi:  dirs $(BUILD_DIR)/srhn_multisem.o
