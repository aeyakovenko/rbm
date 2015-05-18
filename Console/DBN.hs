{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes       #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeFamilies      #-}
module Console.DBN where

import Yesod
--import Control.Concurrent(forkIO)


data Console = Console
mkYesod "Console" [parseRoutes|
/ ConsoleR GET
|]
instance Yesod Console

getConsoleR :: Handler Html
getConsoleR = defaultLayout [whamlet|Hello World!|]

main :: IO ()
main = warp 8000 Console

