
hs_files=RBM.hs
cabal_files=rbm.cabal

dist/cabal.test.ok:dist/cabal.build.ok
	cabal test 2>&1
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config
	cabal build 2>&1
	@touch $@

clean:
	cabal clean

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies --enable-executable-profiling --enable-library-profiling
	cabal configure --enable-tests --enable-coverage
	@touch $@

$$%:;@$(call true)$(info $(call or,$$$*))
