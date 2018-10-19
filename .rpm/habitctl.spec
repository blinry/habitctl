%define debug_package %{nil}

Name:           habitctl
Version:        0.1.0
Release:        1%{?dist}
Summary:        Minimalist command line tool you can use to track and examine your habits.

Group:          Applications/Productivity
License:        GPLv2+
URL:            https://github.com/blinry/habitctl
Source0:        https://github.com/blinry/%{name}/archive/%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root

BuildRequires:  cargo

%description
habitctl is a minimalist command line tool you can use to track and examine your habits. It was born when I grew frustrated with tracking my habits in plain text files using hand-drawn ASCII tables. habitctl tries to get the job done and then get out of your way.

%prep
%setup -q

%build
cargo build --release

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}/usr/bin
cp -a target/release/%{name} %{buildroot}/usr/bin

%clean
rm -rf %{buildroot}

%files
%doc README.md
%defattr(-,root,root,-)
%{_bindir}/*


%changelog
* Sat Oct 20 2018 Sergey Korolev <korolev.srg@gmail.com>
- Initial package for fedora
