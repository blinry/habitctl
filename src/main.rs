extern crate chrono;
#[macro_use]
extern crate clap;
extern crate rprompt;

use chrono::prelude::*;
use clap::{Arg, SubCommand};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let matches = app_from_crate!()
        .subcommand(SubCommand::with_name("ask").about("Ask for status of all habits for today"))
        .subcommand(SubCommand::with_name("log").about("Print habit log"))
        .get_matches();

    let tick = Tick::new();

    match matches.subcommand() {
        ("log", Some(sub_matches)) => tick.log(),
        _ => {
            // no subcommand used, or "ask"
            tick.ask();
            tick.log();
        }
    }

    //tick.entry("Geatmet", "true");
}

struct Tick {
    habits_file: PathBuf,
    log_file: PathBuf,
}

impl Tick {
    fn new() -> Tick {
        let mut tick_dir = env::home_dir().unwrap();
        tick_dir.push(".tick");
        if !tick_dir.is_dir() {
            println!("Creating {:?}", tick_dir);
            fs::create_dir(&tick_dir).unwrap();
        }

        let mut habits_file = tick_dir.clone();
        habits_file.push("habits");
        if !habits_file.is_file() {
            fs::File::create(&habits_file).unwrap();
        }

        let mut log_file = tick_dir.clone();
        log_file.push("log");
        if !log_file.is_file() {
            fs::File::create(&log_file).unwrap();
        }

        Tick {
            habits_file,
            log_file,
        }
    }

    fn entry(&self, entry: &Entry) {
        let file = OpenOptions::new()
            .append(true)
            .open(&self.log_file)
            .unwrap();

        let last_date = self.last_date();

        if let Some(last_date) = last_date {
            if last_date != entry.date {
                write!(&file, "\n");
            }
        }

        write!(
            &file,
            "{}\t{}\t{}\n",
            &entry.date, &entry.habit, &entry.value
        ).unwrap();
    }

    fn log(&self) {
        let habits = self.get_log();

        let to = Local::now();
        let from = to.checked_sub_signed(chrono::Duration::days(30)).unwrap();

        print!("{0: >20}: ", "");
        let mut current = from;
        while current <= to {
            print!("{0:0>2} ", current.day());
            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
        println!();

        for habit in habits.keys() {
            print!("{0: >20}: ", &habit);

            let mut current = from;
            while current <= to {
                let mut iter = habits.get(habit).unwrap().iter();

                let other_date = format!("{}", current.format("%F"));
                let entry =
                    if let Some((date, value)) = iter.find(|(date, value)| date == &other_date) {
                        if value == "y" {
                            "+"
                        } else if value == "n" {
                            "-"
                        } else {
                            "?"
                        }
                    } else {
                        " "
                    };

                print!("{0: >2} ", &entry);

                current = current
                    .checked_add_signed(chrono::Duration::days(1))
                    .unwrap();
            }
            println!();
        }
    }

    fn ask(&self) {
        let f = File::open(&self.habits_file).unwrap();
        let file = BufReader::new(&f);

        let log = self.get_log();

        let entry_date = format!("{}", Local::now().format("%F"));

        for line in file.lines() {
            let habit = line.unwrap();

            if habit.chars().next().unwrap() == '#' {
                continue;
            }

            if log.contains_key(&habit) {
                let mut iter = log.get(&habit).unwrap().iter();

                if let Some((date, value)) = iter.find(|(date, value)| date == &entry_date) {
                    continue;
                }
            }

            let l = format!("{}? [y/n/-] ", &habit);

            let date = Local::now().format("%F");

            let mut value = String::from("");
            loop {
                value = String::from(rprompt::prompt_reply_stdout(&l).unwrap());
                value = value.trim_right().to_string();

                if value == "y" || value == "n" || value == "" {
                    break;
                }
            }

            if value != "" {
                self.entry(&Entry {
                    date: date.to_string(),
                    habit,
                    value,
                });
            }
        }
    }

    fn get_entries(&self) -> Vec<Entry> {
        let f = File::open(&self.log_file).unwrap();
        let file = BufReader::new(&f);

        let mut entries = vec![];

        for line in file.lines() {
            let l = line.unwrap();
            if l == "" {
                continue;
            }
            let split = l.split("\t");
            let parts: Vec<&str> = split.collect();

            let entry = Entry {
                date: parts[0].to_string(),
                habit: parts[1].to_string(),
                value: parts[2].to_string(),
            };

            entries.push(entry);
        }

        entries
    }

    fn get_log(&self) -> HashMap<String, Vec<(String, String)>> {
        let mut habits = HashMap::new();

        for entry in self.get_entries() {
            if !habits.contains_key(&entry.habit) {
                habits.insert(entry.habit.clone(), vec![]);
            }
            habits
                .get_mut(&entry.habit)
                .unwrap()
                .push((entry.date, entry.value));
        }

        habits
    }

    fn last_date(&self) -> Option<String> {
        match self.get_entries().last() {
            Some(entry) => Some(entry.date.clone()),
            None => None,
        }
    }

    /*
    fn habits() ->  {
        let h = ["a", "b"];
        h.iter();
    }
    */
}

struct Entry {
    date: String,
    habit: String,
    value: String,
}
