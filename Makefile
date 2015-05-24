all:mnist

cabal_files=rbm.cabal
tix_files=perf-repa-RBM.tix

hs_files=RBM/List.hs\
			RBM/Repa.hs\
			RBM/Proto.hs\
			Data/Mnist.hs

tix_files=perf-repa-RBM.tix\
			 trainbatches.tix\
			 testbatches.tix\
			 mnist-DBN.tix\
			 test-repa-DBN.tix\
			 test-repa-RBM.tix\
			 bigtrainbatches.tix\
			 test-DBN.tix\
			 console.tix

all:dist/cabal.test.ok dist/cabal.build.ok

dist/cabal.test.ok:$(hs_files) dist/setup-config tix
	cabal test 2>&1
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config tix
	cabal build 2>&1
	@touch $@

clean:tix
	cabal clean

mnist:tix
	cabal build mnist-DBN
	./dist/build/mnist-DBN/mnist-DBN +RTS -N

console:tix
	cabal build console

batches:tix
	cabal build testbatches
	cabal build trainbatches
	rm dist/test* || echo ok
	rm dist/train* || echo ok
	./dist/build/trainbatches/trainbatches
	rm -f $(tix_files)
	./dist/build/testbatches/testbatches
	rm -f $(tix_files)

tix:
	rm -f $(tix_files)

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies
	cabal configure --enable-coverage --enable-tests
	@touch $@

data: train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

train-images-idx3-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

$$%:;@$(call true)$(info $(call or,$$$*))
