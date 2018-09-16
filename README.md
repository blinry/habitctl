# habitctl

**habitctl** is a minimalist command line tool you can use to track and examine your habits. It was born when I grew frustrated with tracking my habits in plain text files using hand-drawn ASCII tables. habitctl tries to get the job done and then get out of your way.

## Installation

habitctl is written in the Rust programming language, so you'll need a working [Rust installation](https://www.rust-lang.org). You'll probably want to run these commands:

    $ curl -f https://sh.rustup.rs > rust.sh
    $ sh rust.sh
    $ source ~/.cargo/env

Then, compiling Timelens is as easy as this:

    $ git clone https://github.com/timelens/timelens
    $ cd timelens
    $ cargo build --release

This will create the binary `target/release/habitctl`, which you can add to your `$PATH`. Additionally, I like to set up an alias called `h`.

## Usage

When you run `habitctl` for the first time, it will set up the required files:

    $ h
    Welcome to habitctl!
    
    Created /home/seb/.habitctl/habits. This file will list your currently tracked habits.
    Created /home/seb/.habitctl/log. This file will contain your habit log.
    
    You don't have any habits set up!
    Run `habitctl edith` to modify the habit list using your default $EDITOR.
    Then, run `habitctl`! Happy tracking!

Run `h edith` and change the content of the habits file, for example like this:

    # The numbers specifies how often you want to do a habit:
    # 1 means daily, 7 means weekly, 0 means you're just tracking the habit. Some examples:

    1 Meditated
    7 Cleaned the apartment
    0 Had a headache
    1 Used habitctl

Here are some more ideas of what to track:

- got up at a certain time
- used a space repetition software like Anki
- took a multivitamin
- cleared my email inbox
- answered to all texts
- visited and read all Slack workspaces
- practised a language
- self reflection/used a diary
- autogenic training
- published something on my blog/homepage
- worked on a project
- did the dishes
- tidied the apartment
- closed all browser tabs
- tracked caloric intake
- happy
- flow
- relaxed
- coffein intake
- left my comfort zone
- thanked someone

Then, simply run `h` regularly, specify whether or not you did the habit, and get pretty graphs!

    $ h
    2018-09-15+02:00:
                    Meditated ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••[y/n/-] y
        Cleaned the apartment ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••[y/n/-] y
               Had a headache ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••[y/n/-] n
                Used habitctl ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••[y/n/-] y
    2018-09-16+02:00:
                    Meditated •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••━•[y/n/-] y
        Cleaned the apartment •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••━•[y/n/-] n
               Had a headache ••••••••••••••••••••••••••••••••••••••••••••••••••••••••••• •[y/n/-] n
                Used habitctl •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••━•[y/n/-] y


(Some weeks later)

    $ h log
                              ▄▃▃▄▄▃▄▆▆▆▅▆▆▇▆▄▃▄▆▃▆▃▆▂▅▄▃▄▅▆▅▃▃▃▆▂▄▅▄▅▅▅▆▄▄▆▇▆▅▅▄▃▅▆▄▆▃▃▂▅▆
                    Meditated ━       ━ ━  ━━         ━    ━   ━ ━   ━━━━━━━━━━━   ━ ━   ━━
        Cleaned the apartment ━──────                 ━──────           ━──────            
               Had a headache             ━  ━     ━━                  ━━   ━   ━━         
                Used habitctl    ━ ━━━ ━  ━━━   ━ ━ ━       ━ ━ ━  ━ ━ ━━ ━ ━ ━━━━   ━    ━

                                             ... some habits omitted ...

    Yesterday's score: 73%

The score specifies how many of the due habits you did that day. The sparkline at the top give a graphical representation of the score. The thick lines in the graph say that you did the habit, the thin lines say that that it was okay that you didn't to it.

Enter `h help` if you're lost:

    $ h help
    USAGE:
        habitctl [SUBCOMMAND]
    
    FLAGS:
        -h, --help       Prints help information
        -V, --version    Prints version information
    
    SUBCOMMANDS:
        ask      Ask for status of all habits for a day
        edit     Edit habit log file
        edith    Edit list of current habits
        help     Prints this message or the help of the given subcommand(s)
        log      Print habit log
        todo     Print unresolved tasks for today

## License: GPLv2+

*habitctl* is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
