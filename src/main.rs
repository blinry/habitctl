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
        .subcommand(
            SubCommand::with_name("ask")
                .about("Ask for status of all habits for a day")
                .arg(Arg::with_name("days ago").index(1).default_value("0")),
        )
        .subcommand(SubCommand::with_name("log").about("Print habit log"))
        .subcommand(SubCommand::with_name("todo").about("Print unresolved tasks for today"))
        .get_matches();

    let mut tick = Tick::new();
    tick.load();

    match matches.subcommand() {
        ("log", Some(sub_matches)) => tick.log(),
        ("todo", Some(sub_matches)) => tick.todo(),
        ("ask", Some(sub_matches)) => {
            let ago: i64 = sub_matches.value_of("days ago").unwrap().parse().unwrap();
            tick.ask(ago);
            tick.log();
        }
        _ => {
            // no subcommand used
            tick.ask(0);
            tick.log();
        }
    }

    //tick.entry("Geatmet", "true");
}

struct Tick {
    habits_file: PathBuf,
    log_file: PathBuf,
    habits: Vec<String>,
    log: HashMap<String, Vec<(String, String)>>,
    entries: Vec<Entry>,
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
            habits: vec![],
            log: HashMap::new(),
            entries: vec![],
        }
    }

    fn load(&mut self) {
        self.log = self.get_log();
        self.habits = self.get_habits();
        self.entries = self.get_entries();
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
        let to = Local::now();
        let from = to.checked_sub_signed(chrono::Duration::days(20)).unwrap();

        print!("{0: >25}: ", "");
        let mut current = from;
        while current <= to {
            print!("{0:0>2} ", current.day());
            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
        println!();

        for habit in self.log.keys() {
            print!("{0: >25}: ", &habit);

            let mut current = from;
            while current <= to {
                let mut iter = self.log.get(habit).unwrap().iter();

                let other_date = format!("{}", current.format("%F"));
                let entry =
                    if let Some((date, value)) = iter.find(|(date, value)| date == &other_date) {
                        if value == "y" {
                            "+"
                        } else if value == "n" {
                            " "
                        } else {
                            "."
                        }
                    } else {
                        "?"
                    };

                print!("{0: >2} ", &entry);

                current = current
                    .checked_add_signed(chrono::Duration::days(1))
                    .unwrap();
            }
            println!();
        }

        let date = format!("{}", to.checked_sub_signed(chrono::Duration::days(1)).unwrap().format("%F"));
        println!("Yesterday's score: {}%", self.get_score(&date));
    }

    fn get_habits(&self) -> Vec<String> {
        let f = File::open(&self.habits_file).unwrap();
        let file = BufReader::new(&f);

        let mut habits = vec![];

        for line in file.lines() {
            let habit = line.unwrap();
            if habit.chars().next().unwrap() != '#' {
                habits.push(habit);
            }
        }

        habits
    }

    fn ask(&self, ago: i64) {
        let entry_date = format!(
            "{}",
            Local::now()
                .checked_sub_signed(chrono::Duration::days(ago))
                .unwrap()
                .format("%F")
        );

        println!("{}:", &entry_date);

        for habit in self.get_todo(&entry_date) {
            let l = format!("{}? [y/n/-] ", &habit);

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
                    date: entry_date.clone(),
                    habit,
                    value,
                });
            }
        }
    }

    fn todo(&self) {
        let entry_date = format!(
            "{}",
            Local::now()
                //.checked_sub_signed(chrono::Duration::days(ago))
                //.unwrap()
                .format("%F")
        );

        for habit in self.get_todo(&entry_date) {
            println!("{}", &habit);
        }
    }

    fn get_todo(&self, todo_date: &String) -> Vec<String> {
        let mut habits = self.get_habits();

        habits.retain(|h| {
            if self.log.contains_key(&h.clone()) {
                let mut iter = self.log.get(&h.clone()).unwrap().iter();

                if let Some((date, value)) = iter.find(|(date, value)| date == todo_date) {
                    return false;
                }
            }
            return true;
        });

        habits
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
        let mut log = HashMap::new();

        for entry in self.get_entries() {
            if !log.contains_key(&entry.habit) {
                log.insert(entry.habit.clone(), vec![]);
            }
            log
                .get_mut(&entry.habit)
                .unwrap()
                .push((entry.date, entry.value));
        }

        log
    }

    fn last_date(&self) -> Option<String> {
        self.get_entries()
            .last()
            .and_then(|entry| Some(entry.date.clone()))
    }

    fn get_score(&self, score_date: &String) -> f32 {
        let todo = self.get_todo(&score_date);

        100.0 - 100.0 * todo.len() as f32 / self.habits.len() as f32
    }
}

struct Entry {
    date: String,
    habit: String,
    value: String,
}
