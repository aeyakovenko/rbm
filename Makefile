hs_files=RBM/List.hs RBM/Repa.hs RBM/Proto.hs
cabal_files=rbm.cabal
tix_files=perf-repa-RBM.tix

all:dist/cabal.test.ok dist/cabal.build.ok

dist/cabal.test.ok:$(hs_files) dist/setup-config
	rm -f $(tix_files)
	cabal test 2>&1
	rm -f $(tix_files)
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config
	cabal build 2>&1
	@touch $@

clean:
	rm -f $(tix_files)
	cabal clean

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies
	cabal configure --enable-coverage --enable-tests
	@touch $@

mnist.pkl.gz:
	wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

$$%:;@$(call true)$(info $(call or,$$$*))
