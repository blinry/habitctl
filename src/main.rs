extern crate chrono;
#[macro_use]
extern crate clap;
extern crate dirs;
extern crate open;
extern crate rprompt;
extern crate math;

use chrono::prelude::*;
use clap::{Arg, SubCommand};
use std::cmp;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::process::Command;
use math::round;

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
            Local::now()
                .date()
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
            habitctl.log(&vec![]);
        }
        ("edit", Some(_)) => habitctl.edit(),
        ("edith", Some(_)) => habitctl.edith(),
        _ => {
            // no subcommand used
            habitctl.assert_habits();
            habitctl.ask(ago);
            habitctl.log(&vec![]);
        }
    }
}

struct HabitCtl {
    habits_file: PathBuf,
    log_file: PathBuf,
    habits: Vec<Habit>,
    log: HashMap<String, Vec<(Date<Local>, String)>>,
    entries: Vec<Entry>,
}

#[derive(PartialEq)]
enum DayStatus {
    Unknown,
    NotDone,
    Done,
    Satisfied,
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

        let mut log_file = habitctl_dir.clone();
        log_file.push("log");
        if !log_file.is_file() {
            fs::File::create(&log_file).unwrap();

            let file = OpenOptions::new().append(true).open(&habits_file).unwrap();
            write!(
                &file,
                "# The numbers specifies how often you want to do a habit:\n"
            );
            write!(
                &file,
                "# 1 means daily, 7 means weekly, 0 means you're just tracking the habit. Some examples:\n"
            );
            write!(
                &file,
                "\n# 1 Meditated\n# 7 Cleaned the apartment\n# 0 Had a headache\n# 1 Used habitctl\n"
            );

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
                write!(&file, "\n").unwrap();
            }
        }

        write!(
            &file,
            "{}\t{}\t{}\n",
            &entry.date.format("%F"),
            &entry.habit,
            &entry.value
        ).unwrap();
    }

    fn log(&self, filters: &Vec<&str>) {
        let to = Local::now();
        let from = to.checked_sub_signed(chrono::Duration::days(100)).unwrap();

        print!("{0: >25} ", "");
        let mut current = from;
        while current <= to {
            print!("{0: >1}", self.spark(self.get_score(&current.date())));
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

            self.print_habit_row(&habit, from.date(), to.date());
            println!();
        }

        if !self.habits.is_empty() {
            let date = to
                .checked_sub_signed(chrono::Duration::days(1))
                .unwrap()
                .date();
            println!("Yesterday's score: {}%", self.get_score(&date));
        }
    }

    fn print_habit_row(&self, habit: &Habit, from: Date<Local>, to: Date<Local>) {
        print!("{0: >25} ", habit.name);

        let mut current = from;
        while current <= to {
            print!(
                "{0: >1}",
                &self.status_to_symbol(&self.day_status(&habit, &current))
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
        let from = Local::now()
            .checked_sub_signed(chrono::Duration::days(ago))
            .unwrap()
            .date();

        let to = Local::now();
        let log_from = to.checked_sub_signed(chrono::Duration::days(60)).unwrap();

        let now = Local::now().date();

        let mut current = from;
        while current <= now {
            if !self.get_todo(&current).is_empty() {
                println!("{}:", &current);
            }

            for habit in self.get_todo(&current) {
                self.print_habit_row(&habit, log_from.date(), current.clone());
                let l = format!("[y/n/-] ");

                let mut value;
                loop {
                    value = String::from(rprompt::prompt_reply_stdout(&l).unwrap());
                    value = value.trim_right().to_string();

                    if value == "y" || value == "n" || value == "" {
                        break;
                    }
                }

                if value != "" {
                    self.entry(&Entry {
                        date: current.clone(),
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
        let entry_date = Local::now().date();

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

    fn get_todo(&self, todo_date: &Date<Local>) -> Vec<Habit> {
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
            if l == "" {
                continue;
            }
            let split = l.split('\t');
            let parts: Vec<&str> = split.collect();

            let date_str = format!("{}T00:00:00+00:00", parts[0]);

            let entry = Entry {
                date: DateTime::parse_from_rfc3339(&date_str)
                    .unwrap()
                    .with_timezone(&Local)
                    .date(),
                habit: parts[1].to_string(),
                value: parts[2].to_string(),
            };

            entries.push(entry);
        }

        entries
    }

    fn get_entry(&self, date: &Date<Local>, habit: &String) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|entry| entry.date == *date && entry.habit == *habit)
    }

    fn day_status(&self, habit: &Habit, date: &Date<Local>) -> DayStatus {
        if let Some(entry) = self.get_entry(&date, &habit.name) {
            if entry.value == "y" {
                DayStatus::Done
            } else if self.habit_satisfied(habit, &date) {
                DayStatus::Satisfied
            } else {
                DayStatus::NotDone
            }
        } else {
            DayStatus::Unknown
        }
    }

    fn status_to_symbol(&self, status: &DayStatus) -> String {
        let symbol = match status {
            DayStatus::Unknown => "•",
            DayStatus::NotDone => " ",
            DayStatus::Done => "━",
            DayStatus::Satisfied => "─",
        };
        String::from(symbol)
    }

    fn habit_satisfied(&self, habit: &Habit, date: &Date<Local>) -> bool {
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

    fn get_log(&self) -> HashMap<String, Vec<(Date<Local>, String)>> {
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

    fn first_date(&self) -> Option<Date<Local>> {
        self.get_entries()
            .first()
            .and_then(|entry| Some(entry.date.clone()))
    }

    fn last_date(&self) -> Option<Date<Local>> {
        self.get_entries()
            .last()
            .and_then(|entry| Some(entry.date.clone()))
    }

    fn get_score(&self, score_date: &Date<Local>) -> f32 {
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
                let status = self.day_status(&habit, &score_date);
                habit.every_days > 0
                    && (status == DayStatus::Done || status == DayStatus::Satisfied)
            })
            .collect();
        done.retain(|value| *value);

        if !todo.is_empty() {
            round::ceil((100.0 * done.len() as f32 / todo.len() as f32).into(), 1) as f32
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

    fn open_file(&self, filename: &PathBuf) {
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
    date: Date<Local>,
    habit: String,
    value: String,
}

struct Habit {
    every_days: i64,
    name: String,
}
