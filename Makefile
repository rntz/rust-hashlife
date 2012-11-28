hlife: hlife.rc FORCE
	rustc -g -L . $<

.PHONY: clean FORCE
clean:
	rm -f hlife
