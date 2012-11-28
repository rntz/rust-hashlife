hlife: hlife.rc hlife.rs parse.rs
	rustc -g -L . $<

.PHONY: clean
clean:
	rm -f hlife
