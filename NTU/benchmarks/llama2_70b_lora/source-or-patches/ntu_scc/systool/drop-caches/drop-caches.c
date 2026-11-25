/*
 * drop-caches.c - setuid-root launcher for /mnt/opt/systools/drop-caches.sh
 *
 * Same shape as the set-cpu-freq / set-gpu-freq wrappers: it does not do the
 * work itself, it only acquires root safely and runs the root-owned script
 * with a clean environment. All policy lives in the script.
 *
 * Checks before executing anything:
 *   - the script is a regular file, owned by root, not writable by group/others
 *   - setgroups/setgid/setuid to root succeed (i.e. we really are setuid root)
 * Environment passed to the script: PATH, IFS and the caller's numeric uid
 * (DROP_CACHES_CALLER) for the syslog record. Nothing else is inherited.
 *
 * Build:   make            (see Makefile; -DSCRIPT_PATH=... overrides the path)
 * Install: see README.md   (root: install -o root -g root -m 4755 drop-caches /mnt/opt/systools/)
 */
#define _GNU_SOURCE
#include <errno.h>
#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef SCRIPT_PATH
#define SCRIPT_PATH "/mnt/opt/systools/drop-caches.sh"
#endif
#define BASH "/usr/bin/bash"
#define TOOL "drop-caches"

static void fail(const char *msg, int use_errno)
{
    if (use_errno)
        fprintf(stderr, TOOL ": %s (%s)\n", msg, strerror(errno));
    else
        fprintf(stderr, TOOL ": %s\n", msg);
    exit(1);
}

int main(int argc, char **argv)
{
    struct stat st;
    uid_t caller = getuid();
    char caller_env[64];
    char *const script_argv[] = { BASH, SCRIPT_PATH, (argc > 1) ? argv[1] : NULL, NULL };

    /* Only the script decides what arguments mean; we forward at most one
     * (e.g. --help or --quiet) and refuse anything that could be a path. */
    if (argc > 2)
        fail("too many arguments (usage: drop-caches [--quiet|--help])", 0);
    if (argc == 2 && (argv[1][0] != '-' || strlen(argv[1]) > 16))
        fail("bad argument (usage: drop-caches [--quiet|--help])", 0);

    if (stat(SCRIPT_PATH, &st) != 0)
        fail("cannot stat " SCRIPT_PATH, 1);
    if (!S_ISREG(st.st_mode))
        fail(SCRIPT_PATH " is not a regular file", 0);
#ifndef ALLOW_NONROOT_SCRIPT
    if (st.st_uid != 0)
        fail(SCRIPT_PATH " must be owned by root", 0);
#endif
    if (st.st_mode & (S_IWGRP | S_IWOTH))
        fail(SCRIPT_PATH " must not be writable by group or others", 0);

    umask(022);
    if (setgroups(0, NULL) != 0)
        fail("setgroups failed", 1);
    if (setgid(0) != 0)
        fail("setgid failed", 1);
    if (setuid(0) != 0)
        fail("setuid failed (is the binary setuid root?)", 1);
    if (geteuid() != 0 || getuid() != 0)
        fail("failed to acquire root", 0);

    snprintf(caller_env, sizeof caller_env, "DROP_CACHES_CALLER=%lu", (unsigned long)caller);
    {
        char *const clean_env[] = { "PATH=/usr/bin:/bin", "IFS= \t\n", caller_env, NULL };
        execve(BASH, script_argv, clean_env);
    }
    fail("execve failed", 1);
    return 1;
}
