hs_files=RBM/List.hs
cabal_files=rbm.cabal

all:dist/cabal.test.ok dist/cabal.perf.ok dist/cabal.build.ok

dist/cabal.perf.ok:$(hs_files) dist/setup-config
	cabal bench 2>&1
	rm -f *.tix
	@touch $@

dist/cabal.test.ok:$(hs_files) dist/setup-config
	cabal test 2>&1
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config
	cabal build 2>&1
	@touch $@

clean:
	rm -f *.tix
	cabal clean

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies
	cabal configure --enable-tests --enable-coverage --enable-library-profiling --enable-benchmarks
	@touch $@

$$%:;@$(call true)$(info $(call or,$$$*))
