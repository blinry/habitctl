extern crate chrono;
#[macro_use]
extern crate clap;
extern crate dirs;
extern crate math;
extern crate open;
extern crate rprompt;

use chrono::prelude::*;
use clap::{Arg, SubCommand};
use math::round;
use std::cmp;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::process::Command;

fn main() {
    let matches = app_from_crate!()
        .template("{bin} {version}\n{author}\n\n{about}\n\nUSAGE:\n    {usage}\n\nFLAGS:\n{flags}\n\nSUBCOMMANDS:\n{subcommands}")
        .subcommand(
            SubCommand::with_name("ask")
                .about("Ask for status of all habits for a day")
                .arg(Arg::with_name("days ago").index(1)),
        )
        .subcommand(
            SubCommand::with_name("log")
                .about("Print habit log")
                .arg(Arg::with_name("filter").index(1).multiple(true)),
        )
        .subcommand(SubCommand::with_name("todo").about("Print unresolved tasks for today"))
        .subcommand(SubCommand::with_name("edit").about("Edit habit log file"))
        .subcommand(SubCommand::with_name("edith").about("Edit list of current habits"))
        .get_matches();

    let mut habitctl = HabitCtl::new();
    habitctl.load();

    let ago: i64 = if habitctl.first_date().is_some() {
        cmp::min(
            7,
            Local::today()
                .naive_local()
                .signed_duration_since(habitctl.first_date().unwrap())
                .num_days(),
        )
    } else {
        1
    };

    match matches.subcommand() {
        ("log", Some(sub_matches)) => {
            habitctl.assert_habits();
            habitctl.assert_entries();
            let filters = if sub_matches.is_present("filter") {
                sub_matches.values_of("filter").unwrap().collect::<Vec<_>>()
            } else {
                vec![]
            };
            habitctl.log(&filters);
        }
        ("todo", Some(_)) => {
            habitctl.assert_habits();
            habitctl.todo()
        }
        ("ask", Some(sub_matches)) => {
            habitctl.assert_habits();
            let ago: i64 = if sub_matches.is_present("days ago") {
                sub_matches.value_of("days ago").unwrap().parse().unwrap()
            } else {
                ago
            };
            habitctl.ask(ago);
            habitctl.log(&[]);
        }
        ("edit", Some(_)) => habitctl.edit(),
        ("edith", Some(_)) => habitctl.edith(),
        _ => {
            // no subcommand used
            habitctl.assert_habits();
            habitctl.ask(ago);
            habitctl.log(&[]);
        }
    }
}

struct HabitCtl {
    habits_file: PathBuf,
    log_file: PathBuf,
    habits: Vec<Habit>,
    log: HashMap<String, Vec<(NaiveDate, String)>>,
    entries: Vec<Entry>,
}

#[derive(PartialEq)]
enum DayStatus {
    Unknown,
    NotDone,
    Done,
    Satisfied,
    Skipped,
    Skipified,
    Warning,
}

impl HabitCtl {
    fn new() -> HabitCtl {
        let mut habitctl_dir = dirs::home_dir().unwrap();
        habitctl_dir.push(".habitctl");
        if !habitctl_dir.is_dir() {
            println!("Welcome to habitctl!\n");
            fs::create_dir(&habitctl_dir).unwrap();
        }

        let mut habits_file = habitctl_dir.clone();
        habits_file.push("habits");
        if !habits_file.is_file() {
            fs::File::create(&habits_file).unwrap();
            println!(
                "Created {}. This file will list your currently tracked habits.",
                habits_file.to_str().unwrap()
            );
        }

        let mut log_file = habitctl_dir;
        log_file.push("log");
        if !log_file.is_file() {
            fs::File::create(&log_file).unwrap();

            let file = OpenOptions::new().append(true).open(&habits_file).unwrap();
            writeln!(
                &file, "\
                # The numbers specifies how often you want to do a habit:\n\
                # 1 means daily, 7 means weekly, 0 means you're just tracking the habit. Some examples:\n\
                \n\
                # 1 Meditated\n\
                # 7 Cleaned the apartment\n\
                # 0 Had a headache\n\
                # 1 Used habitctl"
            ).unwrap();

            println!(
                "Created {}. This file will contain your habit log.\n",
                log_file.to_str().unwrap()
            );
        }

        HabitCtl {
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
                writeln!(&file).unwrap();
            }
        }

        writeln!(
            &file,
            "{}\t{}\t{}",
            &entry.date.format("%F"),
            &entry.habit,
            &entry.value
        )
        .unwrap();
    }

    fn log(&self, filters: &[&str]) {
        let to = Local::today().naive_local();
        let from = to.checked_sub_signed(chrono::Duration::days(100)).unwrap();

        print!("{0: >25} ", "");
        let mut current = from;
        while current <= to {
            print!("{0: >1}", self.spark(self.get_score(&current)));
            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
        println!();

        for habit in self.habits.iter() {
            let mut show = false;
            for filter in filters.iter() {
                if habit.name.to_lowercase().contains(*filter) {
                    show = true;
                }
            }
            if filters.is_empty() {
                show = true;
            }

            if !show {
                continue;
            }

            self.print_habit_row(habit, from, to);
            println!();
        }

        if !self.habits.is_empty() {
            let date = to.checked_sub_signed(chrono::Duration::days(1)).unwrap();
            println!("Yesterday's score: {}%", self.get_score(&date));
        }
    }

    fn print_habit_row(&self, habit: &Habit, from: NaiveDate, to: NaiveDate) {
        print!("{0: >25} ", habit.name);

        let mut current = from;
        while current <= to {
            print!(
                "{0: >1}",
                &self.status_to_symbol(&self.day_status(habit, &current))
            );

            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
    }

    fn get_habits(&self) -> Vec<Habit> {
        let f = File::open(&self.habits_file).unwrap();
        let file = BufReader::new(&f);

        let mut habits = vec![];

        for line in file.lines() {
            let l = line.unwrap();

            if l.chars().count() > 0 {
                let first_char = l.chars().next().unwrap();
                if first_char != '#' && first_char != '\n' {
                    let split = l.trim().splitn(2, ' ');
                    let parts: Vec<&str> = split.collect();

                    habits.push(Habit {
                        every_days: parts[0].parse().unwrap(),
                        name: String::from(parts[1]),
                    });
                }
            }
        }

        habits
    }

    fn ask(&mut self, ago: i64) {
        let from = Local::today()
            .naive_local()
            .checked_sub_signed(chrono::Duration::days(ago))
            .unwrap();

        let to = Local::today().naive_local();
        let log_from = to.checked_sub_signed(chrono::Duration::days(60)).unwrap();

        let now = Local::today().naive_local();

        let mut current = from;
        while current <= now {
            if !self.get_todo(&current).is_empty() {
                println!("{}:", &current);
            }

            for habit in self.get_todo(&current) {
                self.print_habit_row(&habit, log_from, current);
                let l = "[y/n/s/⏎] ";

                let mut value;
                loop {
                    value = rprompt::prompt_reply_stdout(l).unwrap();
                    value = value.trim_end().to_string();

                    if value == "y" || value == "n" || value == "s" || value.is_empty() {
                        break;
                    }
                }

                if !value.is_empty() {
                    self.entry(&Entry {
                        date: current,
                        habit: habit.name,
                        value,
                    });
                }
            }

            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
            self.load();
        }
    }

    fn todo(&self) {
        let entry_date = Local::today().naive_local();

        for habit in self.get_todo(&entry_date) {
            if habit.every_days > 0 {
                println!("{}", &habit.name);
            }
        }
    }

    fn edit(&self) {
        self.open_file(&self.log_file);
    }

    fn edith(&self) {
        self.open_file(&self.habits_file);
    }

    fn get_todo(&self, todo_date: &NaiveDate) -> Vec<Habit> {
        let mut habits = self.get_habits();

        habits.retain(|h| {
            if self.log.contains_key(&h.name.clone()) {
                let mut iter = self.log[&h.name.clone()].iter();
                if iter.any(|(date, _value)| date == todo_date) {
                    return false;
                }
            }
            true
        });

        habits
    }

    fn get_entries(&self) -> Vec<Entry> {
        let f = File::open(&self.log_file).unwrap();
        let file = BufReader::new(&f);

        let mut entries = vec![];

        for line in file.lines() {
            let l = line.unwrap();
            if l.is_empty() {
                continue;
            }
            let split = l.split('\t');
            let parts: Vec<&str> = split.collect();

            let entry = Entry {
                date: NaiveDate::parse_from_str(parts[0], "%Y-%m-%d").unwrap(),
                habit: parts[1].to_string(),
                value: parts[2].to_string(),
            };

            entries.push(entry);
        }

        entries
    }

    fn get_entry(&self, date: &NaiveDate, habit: &str) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|entry| entry.date == *date && entry.habit == *habit)
    }

    fn day_status(&self, habit: &Habit, date: &NaiveDate) -> DayStatus {
        if let Some(entry) = self.get_entry(date, &habit.name) {
            if entry.value == "y" {
                DayStatus::Done
            } else if entry.value == "s" {
                DayStatus::Skipped
            } else if self.habit_satisfied(habit, date) {
                DayStatus::Satisfied
            } else if self.habit_skipified(habit, date) {
                DayStatus::Skipified
            } else {
                DayStatus::NotDone
            }
        } else if self.habit_warning(habit, date) {
            DayStatus::Warning
        } else {
            DayStatus::Unknown
        }
    }

    fn status_to_symbol(&self, status: &DayStatus) -> String {
        let symbol = match status {
            DayStatus::Unknown => " ",
            DayStatus::NotDone => " ",
            DayStatus::Done => "━",
            DayStatus::Satisfied => "─",
            DayStatus::Skipped => "•",
            DayStatus::Skipified => "·",
            DayStatus::Warning => "!",
        };
        String::from(symbol)
    }

    fn habit_satisfied(&self, habit: &Habit, date: &NaiveDate) -> bool {
        if habit.every_days < 1 {
            return false;
        }

        let from = date
            .checked_sub_signed(chrono::Duration::days(habit.every_days - 1))
            .unwrap();
        let mut current = from;
        while current <= *date {
            if let Some(entry) = self.get_entry(&current, &habit.name) {
                if entry.value == "y" {
                    return true;
                }
            }
            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
        false
    }

    fn habit_skipified(&self, habit: &Habit, date: &NaiveDate) -> bool {
        if habit.every_days < 1 {
            return false;
        }

        let from = date
            .checked_sub_signed(chrono::Duration::days(habit.every_days - 1))
            .unwrap();
        let mut current = from;
        while current <= *date {
            if let Some(entry) = self.get_entry(&current, &habit.name) {
                if entry.value == "s" {
                    return true;
                }
            }
            current = current
                .checked_add_signed(chrono::Duration::days(1))
                .unwrap();
        }
        false
    }

    fn habit_warning(&self, habit: &Habit, date: &NaiveDate) -> bool {
        if habit.every_days < 1 {
            return false;
        }

        let from = date
            .checked_sub_signed(chrono::Duration::days(habit.every_days))
            .unwrap();
        let mut current = *date;
        while current >= from {
            if let Some(entry) = self.get_entry(&current, &habit.name) {
                if (entry.value == "y" || entry.value == "s")
                    && current - from > chrono::Duration::days(0)
                {
                    return false;
                } else if (entry.value == "y" || entry.value == "s")
                    && current - from == chrono::Duration::days(0)
                {
                    return true;
                }
            }
            current = current
                .checked_sub_signed(chrono::Duration::days(1))
                .unwrap();
        }
        false
    }

    fn get_log(&self) -> HashMap<String, Vec<(NaiveDate, String)>> {
        let mut log = HashMap::new();

        for entry in self.get_entries() {
            if !log.contains_key(&entry.habit) {
                log.insert(entry.habit.clone(), vec![]);
            }
            log.get_mut(&entry.habit)
                .unwrap()
                .push((entry.date, entry.value));
        }

        log
    }

    fn first_date(&self) -> Option<NaiveDate> {
        self.get_entries().first().map(|entry| entry.date)
    }

    fn last_date(&self) -> Option<NaiveDate> {
        self.get_entries().last().map(|entry| entry.date)
    }

    fn get_score(&self, score_date: &NaiveDate) -> f32 {
        let mut todo: Vec<bool> = self
            .habits
            .iter()
            .map(|habit| habit.every_days > 0)
            .collect();
        todo.retain(|value| *value);

        let mut done: Vec<bool> = self
            .habits
            .iter()
            .map(|habit| {
                let status = self.day_status(habit, score_date);
                habit.every_days > 0
                    && (status == DayStatus::Done || status == DayStatus::Satisfied)
            })
            .collect();
        done.retain(|value| *value);

        let mut skip: Vec<bool> = self
            .habits
            .iter()
            .map(|habit| {
                let status = self.day_status(habit, score_date);
                habit.every_days > 0
                    && (status == DayStatus::Skipped || status == DayStatus::Skipified)
            })
            .collect();
        skip.retain(|value| *value);

        if !todo.is_empty() {
            round::ceil(
                (100.0 * done.len() as f32 / (todo.len() - skip.len()) as f32).into(),
                1,
            ) as f32
        } else {
            0.0
        }
    }

    fn assert_habits(&self) {
        if self.habits.is_empty() {
            println!(
                "You don't have any habits set up!\nRun `habitctl edith` to modify the habit list using your default $EDITOR.");
            println!("Then, run `habitctl`! Happy tracking!");
            process::exit(1);
        }
    }

    fn assert_entries(&self) {
        if self.entries.is_empty() {
            println!("Please run `habitctl`! Happy tracking!");
            process::exit(1);
        }
    }

    fn spark(&self, score: f32) -> String {
        let sparks = vec![" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
        let i = cmp::min(sparks.len() - 1, (score / sparks.len() as f32) as usize);
        String::from(sparks[i])
    }

    fn open_file(&self, filename: &Path) {
        let editor = env::var("EDITOR").unwrap_or_else(|_| String::from("vi"));
        Command::new(editor)
            .arg(filename)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
    }
}

struct Entry {
    date: NaiveDate,
    habit: String,
    value: String,
}

struct Habit {
    every_days: i64,
    name: String,
}
