
hs_files=RBM.hs
cabal_files=rbm.cabal

dist/cabal.test.ok:dist/cabal.build.ok
	cabal test 2>&1
	cabal bench 2>&1
	rm -f perf-RBM.tix
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config
	cabal build 2>&1
	@touch $@

clean:
	rm -f perf-RBM.tix
	cabal clean

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies --enable-executable-profiling --enable-library-profiling
	cabal configure --enable-tests --enable-coverage --enable-executable-profiling --enable-library-profiling
	@touch $@

$$%:;@$(call true)$(info $(call or,$$$*))
